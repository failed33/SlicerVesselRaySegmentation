struct PatchConfig_std430_0
{
    @align(4) depth_0 : u32,
    @align(4) height_0 : u32,
    @align(4) width_0 : u32,
    @align(4) rayCount_0 : u32,
    @align(4) cornerZ_0 : i32,
    @align(4) cornerY_0 : i32,
    @align(4) cornerX_0 : i32,
    @align(4) h0_0 : f32,
};

@binding(0) @group(0) var<storage, read> config_0 : array<PatchConfig_std430_0>;

struct RayGeometry_std430_0
{
    @align(4) originHU_0 : f32,
    @align(4) originZ_0 : i32,
    @align(4) originY_0 : i32,
    @align(4) originX_0 : i32,
    @align(4) stepZ_0 : i32,
    @align(4) stepY_0 : i32,
    @align(4) stepX_0 : i32,
};

@binding(2) @group(0) var<storage, read> ray_geometry_0 : array<RayGeometry_std430_0>;

struct PackedZYX_std430_0
{
    @align(4) z_0 : f32,
    @align(4) y_0 : f32,
    @align(4) x_0 : f32,
};

@binding(5) @group(0) var<storage, read_write> origin_zyx_norm_0 : array<PackedZYX_std430_0>;

@binding(6) @group(0) var<storage, read_write> direction_zyx_norm_0 : array<PackedZYX_std430_0>;

@binding(1) @group(0) var<storage, read> volume_zyx_0 : array<f32>;

@binding(8) @group(0) var<storage, read_write> ray_hu_0 : array<f32>;

@binding(3) @group(0) var<storage, read_write> ray_features_0 : array<f32>;

@binding(7) @group(0) var<storage, read_write> voxel_indices_0 : array<vec2<u32>>;

@binding(4) @group(0) var<storage, read_write> ray_mask_0 : array<u32>;

@binding(9) @group(0) var<storage, read_write> ct_patch_0 : array<f32>;

fn patchCorner_0() -> vec3<i32>
{
    return vec3<i32>(config_0[i32(0)].cornerZ_0, config_0[i32(0)].cornerY_0, config_0[i32(0)].cornerX_0);
}

fn toFloat_0( v_0 : vec3<i32>) -> vec3<f32>
{
    return vec3<f32>(v_0) / vec3<f32>(6.5536e+04f);
}

struct PackedZYX_0
{
     z_0 : f32,
     y_0 : f32,
     x_0 : f32,
};

fn PackedZYX_x24init_0( z_1 : f32,  y_1 : f32,  x_1 : f32) -> PackedZYX_0
{
    var _S1 : PackedZYX_0;
    _S1.z_0 = z_1;
    _S1.y_0 = y_1;
    _S1.x_0 = x_1;
    return _S1;
}

fn writeGeometry_0( ray_0 : u32,  originFP_0 : vec3<i32>,  stepFP_0 : vec3<i32>)
{
    var _S2 : vec3<f32> = vec3<f32>(48.0f);
    var origin_0 : vec3<f32> = (vec3<f32>(((originFP_0 >> (vec3<u32>(u32(16))))) - patchCorner_0()) + vec3<f32>((originFP_0 & (vec3<i32>(i32(65535))))) / vec3<f32>(6.5536e+04f)) / _S2;
    var direction_0 : vec3<f32> = toFloat_0(stepFP_0) / _S2;
    var _S3 : PackedZYX_0 = PackedZYX_x24init_0(origin_0.x, origin_0.y, origin_0.z);
    origin_zyx_norm_0[ray_0].z_0 = _S3.z_0;
    origin_zyx_norm_0[ray_0].y_0 = _S3.y_0;
    origin_zyx_norm_0[ray_0].x_0 = _S3.x_0;
    var _S4 : PackedZYX_0 = PackedZYX_x24init_0(direction_0.x, direction_0.y, direction_0.z);
    direction_zyx_norm_0[ray_0].z_0 = _S4.z_0;
    direction_zyx_norm_0[ray_0].y_0 = _S4.y_0;
    direction_zyx_norm_0[ray_0].x_0 = _S4.x_0;
    return;
}

fn roundFixed_0( value_0 : i32) -> i32
{
    var quotient_0 : i32 = (value_0 >> (u32(16)));
    var remainder_0 : i32 = (value_0 & (i32(65535)));
    if(remainder_0 < i32(32768))
    {
        return quotient_0;
    }
    if(remainder_0 > i32(32768))
    {
        return quotient_0 + i32(1);
    }
    var _S5 : i32;
    if(((quotient_0 & (i32(1)))) == i32(0))
    {
        _S5 = quotient_0;
    }
    else
    {
        _S5 = quotient_0 + i32(1);
    }
    return _S5;
}

fn roundVoxel_0( v_1 : vec3<i32>) -> vec3<i32>
{
    return vec3<i32>(roundFixed_0(v_1.x), roundFixed_0(v_1.y), roundFixed_0(v_1.z));
}

fn dimensions_0() -> vec3<i32>
{
    return vec3<i32>(i32(config_0[i32(0)].depth_0), i32(config_0[i32(0)].height_0), i32(config_0[i32(0)].width_0));
}

fn inVolume_0( v_2 : vec3<i32>) -> bool
{
    var _S6 : bool;
    if((all((v_2 >= vec3<i32>(i32(0))))))
    {
        _S6 = (all((v_2 < (dimensions_0()))));
    }
    else
    {
        _S6 = false;
    }
    return _S6;
}

fn inPatch_0( v_3 : vec3<i32>) -> bool
{
    var _S7 : bool;
    if((all((v_3 >= vec3<i32>(i32(0))))))
    {
        _S7 = (all((v_3 < vec3<i32>(i32(48)))));
    }
    else
    {
        _S7 = false;
    }
    return _S7;
}

fn volumeIndex_0( v_4 : vec3<i32>) -> u32
{
    return u32(v_4.x) * config_0[i32(0)].height_0 * config_0[i32(0)].width_0 + u32(v_4.y) * config_0[i32(0)].width_0 + u32(v_4.z);
}

fn loadClamped_0( v_5 : vec3<i32>) -> f32
{
    return volume_zyx_0[volumeIndex_0(clamp(v_5, vec3<i32>(i32(0)), dimensions_0() - vec3<i32>(i32(1))))];
}

fn sampleHU_0( posFP_0 : vec3<i32>) -> f32
{
    var lo_0 : vec3<i32> = (posFP_0 >> (vec3<u32>(u32(16))));
    var fraction_0 : vec3<f32> = vec3<f32>((posFP_0 & (vec3<i32>(i32(65535))))) / vec3<f32>(6.5536e+04f);
    var _S8 : f32 = fraction_0.z;
    var _S9 : f32 = fraction_0.y;
    return mix(mix(mix(loadClamped_0(lo_0), loadClamped_0(lo_0 + vec3<i32>(i32(0), i32(0), i32(1))), _S8), mix(loadClamped_0(lo_0 + vec3<i32>(i32(0), i32(1), i32(0))), loadClamped_0(lo_0 + vec3<i32>(i32(0), i32(1), i32(1))), _S8), _S9), mix(mix(loadClamped_0(lo_0 + vec3<i32>(i32(1), i32(0), i32(0))), loadClamped_0(lo_0 + vec3<i32>(i32(1), i32(0), i32(1))), _S8), mix(loadClamped_0(lo_0 + vec3<i32>(i32(1), i32(1), i32(0))), loadClamped_0(lo_0 + vec3<i32>(i32(1), i32(1), i32(1))), _S8), _S9), fraction_0.x);
}

fn patchIndex_0( v_6 : vec3<i32>) -> u32
{
    return u32(v_6.x) * u32(48) * u32(48) + u32(v_6.y) * u32(48) + u32(v_6.z);
}

fn writeMask_0( index_0 : u32,  valid_0 : bool)
{
    var word_0 : u32 = index_0 / u32(4);
    var _S10 : u32 = index_0 % u32(4);
    if(_S10 == u32(0))
    {
        ray_mask_0[word_0] = u32(0);
    }
    ray_mask_0[word_0] = (ray_mask_0[word_0] | (((u32(valid_0) << ((_S10 * u32(8)))))));
    return;
}

fn buildRay_0( ray_1 : u32)
{
    var hu_0 : f32;
    var geometry_0 : RayGeometry_std430_0 = ray_geometry_0[ray_1];
    var originFP_1 : vec3<i32> = vec3<i32>(geometry_0.originZ_0, geometry_0.originY_0, geometry_0.originX_0);
    var stepFP_1 : vec3<i32> = vec3<i32>(geometry_0.stepZ_0, geometry_0.stepY_0, geometry_0.stepX_0);
    var _S11 : f32 = max(abs(geometry_0.originHU_0), 1.0f);
    var _S12 : vec3<i32> = patchCorner_0();
    writeGeometry_0(ray_1, originFP_1, stepFP_1);
    var stopped_0 : bool = false;
    var k_0 : u32 = u32(0);
    var validCount_0 : u32 = u32(0);
    for(;;)
    {
        if(k_0 < u32(120))
        {
        }
        else
        {
            break;
        }
        var _S13 : u32 = k_0 + u32(1);
        var posFP_1 : vec3<i32> = originFP_1 + stepFP_1 * vec3<i32>(i32(_S13));
        var voxel_0 : vec3<i32> = roundVoxel_0(posFP_1);
        var local_0 : vec3<i32> = voxel_0 - _S12;
        var _S14 : bool;
        if(!stopped_0)
        {
            _S14 = inVolume_0(voxel_0);
        }
        else
        {
            _S14 = false;
        }
        var valid_1 : bool;
        if(_S14)
        {
            valid_1 = inPatch_0(local_0);
        }
        else
        {
            valid_1 = false;
        }
        var sample_0 : u32 = ray_1 * u32(120) + k_0;
        if(valid_1)
        {
            hu_0 = sampleHU_0(posFP_1);
        }
        else
        {
            hu_0 = 0.0f;
        }
        ray_hu_0[sample_0] = hu_0;
        ray_features_0[sample_0 * u32(2)] = hu_0 / _S11;
        var _S15 : u32;
        if(valid_1)
        {
            _S15 = patchIndex_0(local_0);
        }
        else
        {
            _S15 = u32(0);
        }
        voxel_indices_0[sample_0] = vec2<u32>(_S15, u32(0));
        writeMask_0(sample_0, valid_1);
        if(valid_1)
        {
            validCount_0 = validCount_0 + u32(1);
        }
        else
        {
            stopped_0 = true;
        }
        k_0 = _S13;
    }
    var _S16 : f32 = max(f32(validCount_0) - 1.0f, 1.0f);
    k_0 = u32(0);
    for(;;)
    {
        if(k_0 < u32(120))
        {
        }
        else
        {
            break;
        }
        if(k_0 < validCount_0)
        {
            hu_0 = f32(k_0) / _S16;
        }
        else
        {
            hu_0 = 0.0f;
        }
        ray_features_0[(ray_1 * u32(120) + k_0) * u32(2) + u32(1)] = hu_0;
        k_0 = k_0 + u32(1);
    }
    return;
}

fn buildPatch_0( index_1 : u32)
{
    var z_2 : u32 = index_1 / u32(2304);
    var rem_0 : u32 = index_1 - z_2 * u32(48) * u32(48);
    var y_2 : u32 = rem_0 / u32(48);
    var voxel_1 : vec3<i32> = patchCorner_0() + vec3<i32>(i32(z_2), i32(y_2), i32(rem_0 - y_2 * u32(48)));
    var value_1 : f32;
    if(inVolume_0(voxel_1))
    {
        value_1 = volume_zyx_0[volumeIndex_0(voxel_1)];
    }
    else
    {
        value_1 = 0.0f;
    }
    var _S17 : PatchConfig_std430_0 = config_0[i32(0)];
    if((_S17.h0_0) > 9.99999997475242708e-07f)
    {
        value_1 = value_1 / _S17.h0_0;
    }
    ct_patch_0[index_1] = value_1;
    return;
}

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S18 : PatchConfig_std430_0 = config_0[i32(0)];
    var _S19 : u32 = id_0.x;
    if(_S19 < (_S18.rayCount_0))
    {
        buildRay_0(_S19);
        return;
    }
    var patchTask_0 : u32 = _S19 - _S18.rayCount_0;
    if(patchTask_0 < u32(110592))
    {
        buildPatch_0(patchTask_0);
    }
    return;
}

