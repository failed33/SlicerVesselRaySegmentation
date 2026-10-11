// Experimental ModelO patch-input builder.
//
// This shader is intentionally not part of VesselPathTracer/CppCore. It emits
// the tensors that ModelO consumes, rather than the legacy wall/interior maps.
//
// Ray geometry contract:
//   The host supplies each ray as Q16 fixed-point origin and per-sample step in
//   ZYX voxel coordinates. The shader advances samples in integer fixed-point
//   and rounds those fixed coordinates to voxels with half-to-even semantics.
//   This is required because the voxel scatter index is a model input: f32 CPU
//   and GPU recomputation can otherwise disagree by one voxel near .5 ties.
//
// TODO(model-o-webgpu): this still builds per-patch ray tensors for the current
// ONNX graph. The larger speedup is to retrain/export a graph that pre-encodes
// rays once per case and reuses that representation across patch centers.
//
// Bindings:
//   0 cfg_u32: [dim_z, dim_y, dim_x, num_patch_rays, rays_oversample,
//               fixed_point_scale]
//   1 cfg_f32: [spacing_z, spacing_y, spacing_x, step_mm, max_mm,
//               patch_h0, corner_z, corner_y, corner_x]
//               spacing/step/max are carried for host-side parity; this shader
//               consumes the precomputed fixed-point ray lattice directly.
//   2 volume_zyx: flat float32 volume in ZYX row-major order
//   3 ray_origin_hu: per patch ray origin HU
//   4 ray_lattice_fp: per patch ray [origin_z_fp, origin_y_fp, origin_x_fp,
//                                    step_z_fp, step_y_fp, step_x_fp]
//   5 ray_features: output f32, (n_rays, 120, 7)
//   6 ray_mask_u32: output u32, (n_rays, 120), 1 means valid for ModelO scatter
//   7 origin_zyx_norm: output f32, (n_rays, 3)
//   8 direction_zyx_norm: output f32, (n_rays, 3)
//   9 voxel_indices_u32: output u32, (n_rays, 120), patch-local flat index
//  10 ray_hu: output f32, (n_rays, 120), signed HU
//  11 ct_patch: output f32, (48, 48, 48), normalized by patch_h0

const RAY_LENGTH: u32 = 120u;
const FEATURE_COUNT: u32 = 7u;
const PATCH_SIZE: u32 = 48u;
const PATCH_VOXELS: u32 = PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;
const WORKGROUP_SIZE: u32 = 64u;
const NEG_INF: f32 = -1.0e30;

@group(0) @binding(0) var<storage, read> cfg_u32: array<u32>;
@group(0) @binding(1) var<storage, read> cfg_f32: array<f32>;
@group(0) @binding(2) var<storage, read> volume_zyx: array<f32>;
@group(0) @binding(3) var<storage, read> ray_origin_hu: array<f32>;
@group(0) @binding(4) var<storage, read> ray_lattice_fp: array<i32>;

@group(0) @binding(5) var<storage, read_write> ray_features: array<f32>;
@group(0) @binding(6) var<storage, read_write> ray_mask_u32: array<u32>;
@group(0) @binding(7) var<storage, read_write> origin_zyx_norm: array<f32>;
@group(0) @binding(8) var<storage, read_write> direction_zyx_norm: array<f32>;
@group(0) @binding(9) var<storage, read_write> voxel_indices_u32: array<u32>;
@group(0) @binding(10) var<storage, read_write> ray_hu: array<f32>;
@group(0) @binding(11) var<storage, read_write> ct_patch: array<f32>;

fn dims_zyx() -> vec3<u32> {
    return vec3<u32>(cfg_u32[0], cfg_u32[1], cfg_u32[2]);
}

fn num_patch_rays() -> u32 {
    return cfg_u32[3];
}

fn fixed_point_scale_i32() -> i32 {
    return i32(cfg_u32[5]);
}

fn fixed_point_scale_f32() -> f32 {
    return f32(fixed_point_scale_i32());
}

fn patch_h0() -> f32 {
    return cfg_f32[5];
}

fn patch_corner_zyx() -> vec3<i32> {
    return vec3<i32>(
        i32(round(cfg_f32[6])),
        i32(round(cfg_f32[7])),
        i32(round(cfg_f32[8])),
    );
}

fn volume_flat_idx(z: u32, y: u32, x: u32) -> u32 {
    let dims = dims_zyx();
    return z * dims.y * dims.z + y * dims.z + x;
}

fn voxel_in_volume(v: vec3<i32>) -> bool {
    let dims = dims_zyx();
    return v.x >= 0 && v.y >= 0 && v.z >= 0
        && v.x < i32(dims.x)
        && v.y < i32(dims.y)
        && v.z < i32(dims.z);
}

fn round_fixed_to_voxel_scalar(v: i32) -> i32 {
    let scale = fixed_point_scale_i32();
    var q = v / scale;
    var r = v % scale;
    if r < 0 {
        q = q - 1;
        r = r + scale;
    }

    let half = scale / 2;
    if r < half {
        return q;
    }
    if r > half {
        return q + 1;
    }
    return select(q + 1, q, (q % 2) == 0);
}

fn rounded_voxel_from_fixed(pos_fp: vec3<i32>) -> vec3<i32> {
    return vec3<i32>(
        round_fixed_to_voxel_scalar(pos_fp.x),
        round_fixed_to_voxel_scalar(pos_fp.y),
        round_fixed_to_voxel_scalar(pos_fp.z),
    );
}

fn fixed_to_float_vec3(v: vec3<i32>) -> vec3<f32> {
    let scale = fixed_point_scale_f32();
    return vec3<f32>(f32(v.x) / scale, f32(v.y) / scale, f32(v.z) / scale);
}

fn load_volume_clamped(z: i32, y: i32, x: i32) -> f32 {
    let dims = dims_zyx();
    let zz = u32(clamp(z, 0, i32(dims.x) - 1));
    let yy = u32(clamp(y, 0, i32(dims.y) - 1));
    let xx = u32(clamp(x, 0, i32(dims.z) - 1));
    return volume_zyx[volume_flat_idx(zz, yy, xx)];
}

fn sample_hu_linear(pos: vec3<f32>) -> f32 {
    let z0f = floor(pos.x);
    let y0f = floor(pos.y);
    let x0f = floor(pos.z);
    let z0 = i32(z0f);
    let y0 = i32(y0f);
    let x0 = i32(x0f);
    let z1 = z0 + 1;
    let y1 = y0 + 1;
    let x1 = x0 + 1;
    let dz = pos.x - z0f;
    let dy = pos.y - y0f;
    let dx = pos.z - x0f;

    let c000 = load_volume_clamped(z0, y0, x0);
    let c001 = load_volume_clamped(z0, y0, x1);
    let c010 = load_volume_clamped(z0, y1, x0);
    let c011 = load_volume_clamped(z0, y1, x1);
    let c100 = load_volume_clamped(z1, y0, x0);
    let c101 = load_volume_clamped(z1, y0, x1);
    let c110 = load_volume_clamped(z1, y1, x0);
    let c111 = load_volume_clamped(z1, y1, x1);

    let c00 = mix(c000, c001, dx);
    let c01 = mix(c010, c011, dx);
    let c10 = mix(c100, c101, dx);
    let c11 = mix(c110, c111, dx);
    let c0 = mix(c00, c01, dy);
    let c1 = mix(c10, c11, dy);
    return mix(c0, c1, dz);
}

fn feature_index(ray: u32, step: u32, feature: u32) -> u32 {
    return (ray * RAY_LENGTH + step) * FEATURE_COUNT + feature;
}

fn sample_index(ray: u32, step: u32) -> u32 {
    return ray * RAY_LENGTH + step;
}

fn patch_flat(local: vec3<i32>) -> u32 {
    let z = u32(clamp(local.x, 0, i32(PATCH_SIZE) - 1));
    let y = u32(clamp(local.y, 0, i32(PATCH_SIZE) - 1));
    let x = u32(clamp(local.z, 0, i32(PATCH_SIZE) - 1));
    return z * PATCH_SIZE * PATCH_SIZE + y * PATCH_SIZE + x;
}

fn local_in_patch(local: vec3<i32>) -> bool {
    return local.x >= 0 && local.y >= 0 && local.z >= 0
        && local.x < i32(PATCH_SIZE)
        && local.y < i32(PATCH_SIZE)
        && local.z < i32(PATCH_SIZE);
}

fn write_ray_geometry(ray: u32, origin: vec3<f32>, step_voxel: vec3<f32>) {
    let corner = vec3<f32>(patch_corner_zyx());
    let origin_local = (origin - corner) / f32(PATCH_SIZE);
    let direction_norm = step_voxel / f32(PATCH_SIZE);
    let base = ray * 3u;
    origin_zyx_norm[base + 0u] = origin_local.x;
    origin_zyx_norm[base + 1u] = origin_local.y;
    origin_zyx_norm[base + 2u] = origin_local.z;
    direction_zyx_norm[base + 0u] = direction_norm.x;
    direction_zyx_norm[base + 1u] = direction_norm.y;
    direction_zyx_norm[base + 2u] = direction_norm.z;
}

fn load_ray_origin_fp(ray: u32) -> vec3<i32> {
    let base = ray * 6u;
    return vec3<i32>(
        ray_lattice_fp[base + 0u],
        ray_lattice_fp[base + 1u],
        ray_lattice_fp[base + 2u],
    );
}

fn load_ray_step_fp(ray: u32) -> vec3<i32> {
    let base = ray * 6u;
    return vec3<i32>(
        ray_lattice_fp[base + 3u],
        ray_lattice_fp[base + 4u],
        ray_lattice_fp[base + 5u],
    );
}

fn build_ray(ray: u32) {
    var hu_values: array<f32, RAY_LENGTH>;
    var valid_values: array<u32, RAY_LENGTH>;

    let origin_fp = load_ray_origin_fp(ray);
    let step_fp = load_ray_step_fp(ray);
    let origin = fixed_to_float_vec3(origin_fp);
    let step_voxel = fixed_to_float_vec3(step_fp);
    let origin_hu = ray_origin_hu[ray];
    let h0 = max(abs(origin_hu), 1.0);
    let corner = patch_corner_zyx();

    write_ray_geometry(ray, origin, step_voxel);

    var valid_count: u32 = 0u;
    var total_sum: f32 = 0.0;
    var stopped: bool = false;

    for (var k = 0u; k < RAY_LENGTH; k = k + 1u) {
        let pos_fp = origin_fp + step_fp * i32(k + 1u);
        let voxel = rounded_voxel_from_fixed(pos_fp);
        let raw_valid = (!stopped) && voxel_in_volume(voxel);
        let idx = sample_index(ray, k);

        if raw_valid {
            let pos = fixed_to_float_vec3(pos_fp);
            let hu = sample_hu_linear(pos);
            let local = voxel - corner;
            let in_patch = local_in_patch(local);
            let flat = select(0u, patch_flat(local), in_patch);

            hu_values[k] = hu;
            valid_values[k] = 1u;
            ray_hu[idx] = hu;
            voxel_indices_u32[idx] = flat;
            ray_mask_u32[idx] = select(0u, 1u, in_patch);

            valid_count = valid_count + 1u;
            total_sum = total_sum + hu;
        } else {
            stopped = true;
            hu_values[k] = 0.0;
            valid_values[k] = 0u;
            ray_hu[idx] = 0.0;
            voxel_indices_u32[idx] = 0u;
            ray_mask_u32[idx] = 0u;
        }
    }

    if valid_count == 0u {
        for (var k0 = 0u; k0 < RAY_LENGTH; k0 = k0 + 1u) {
            for (var f0 = 0u; f0 < FEATURE_COUNT; f0 = f0 + 1u) {
                ray_features[feature_index(ray, k0, f0)] = 0.0;
            }
        }
        return;
    }

    let n_valid = f32(valid_count);
    let mean = total_sum / n_valid;
    var total_var: f32 = 0.0;
    for (var vk = 0u; vk < RAY_LENGTH; vk = vk + 1u) {
        if valid_values[vk] != 0u {
            let d = hu_values[vk] - mean;
            total_var = total_var + d * d;
        }
    }
    let tv = total_var / n_valid;

    var prefix_sum: f32 = 0.0;
    var running_max: f32 = NEG_INF;
    var previous_hu: f32 = 0.0;
    let len_minus_1 = max(f32(valid_count) - 1.0, 1.0);

    for (var k1 = 0u; k1 < RAY_LENGTH; k1 = k1 + 1u) {
        if valid_values[k1] == 0u {
            for (var f1 = 0u; f1 < FEATURE_COUNT; f1 = f1 + 1u) {
                ray_features[feature_index(ray, k1, f1)] = 0.0;
            }
            continue;
        }

        let hu = hu_values[k1];
        prefix_sum = prefix_sum + hu;
        running_max = max(running_max, hu);

        let idx_f = f32(k1 + 1u);
        let suffix_sum = total_sum - prefix_sum;
        let suffix_count = max(n_valid - idx_f, 1.0);
        let mu_l = prefix_sum / idx_f;
        let mu_r = suffix_sum / suffix_count;
        let raw_eta = idx_f * max(n_valid - idx_f, 0.0) * (mu_l - mu_r) * (mu_l - mu_r);
        let eta2 = select(0.0, min(raw_eta / (n_valid * tv), 1.0), tv > 1.0e-12);
        let grad = select(0.0, (hu - previous_hu) / h0, k1 > 0u);

        ray_features[feature_index(ray, k1, 0u)] = hu / h0;
        ray_features[feature_index(ray, k1, 1u)] = grad;
        ray_features[feature_index(ray, k1, 2u)] = (prefix_sum / idx_f) / h0;
        ray_features[feature_index(ray, k1, 3u)] = (mu_l - mu_r) / h0;
        ray_features[feature_index(ray, k1, 4u)] = eta2;
        ray_features[feature_index(ray, k1, 5u)] = (running_max - hu) / h0;
        ray_features[feature_index(ray, k1, 6u)] = f32(k1) / len_minus_1;

        previous_hu = hu;
    }
}

fn build_ct_patch(voxel_id: u32) {
    let z = voxel_id / (PATCH_SIZE * PATCH_SIZE);
    let rem = voxel_id - z * PATCH_SIZE * PATCH_SIZE;
    let y = rem / PATCH_SIZE;
    let x = rem - y * PATCH_SIZE;
    let corner = patch_corner_zyx();
    let abs_voxel = corner + vec3<i32>(i32(z), i32(y), i32(x));

    var value: f32 = 0.0;
    if voxel_in_volume(abs_voxel) {
        value = volume_zyx[volume_flat_idx(u32(abs_voxel.x), u32(abs_voxel.y), u32(abs_voxel.z))];
    }
    let h0 = patch_h0();
    ct_patch[voxel_id] = select(value, value / h0, h0 > 1.0e-6);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let task = gid.x;
    let ray_count = num_patch_rays();
    if task < ray_count {
        build_ray(task);
        return;
    }

    let patch_task = task - ray_count;
    if patch_task < PATCH_VOXELS {
        build_ct_patch(patch_task);
    }
}
