// ModelO's training-compatible Q16 sampling and seven ONNX input tensors.
// Each invocation owns one complete ray, including its packed mask words.
struct PatchConfig {
    uint depth; uint height; uint width; uint rayCount;
    int cornerZ; int cornerY; int cornerX; float h0;
};
struct RayGeometry {
    float originHU;
    int originZ; int originY; int originX;
    int stepZ; int stepY; int stepX;
};
// Three scalars preserve ONNX's 12-byte row stride. float3 arrays do not.
struct PackedZYX { float z; float y; float x; };

StructuredBuffer<PatchConfig> config;
StructuredBuffer<float> volume_zyx;
StructuredBuffer<RayGeometry> ray_geometry;
RWStructuredBuffer<float> ray_features;
RWStructuredBuffer<uint> ray_mask;
RWStructuredBuffer<PackedZYX> origin_zyx_norm;
RWStructuredBuffer<PackedZYX> direction_zyx_norm;
RWStructuredBuffer<uint2> voxel_indices;
RWStructuredBuffer<float> ray_hu;
RWStructuredBuffer<float> ct_patch;

static const uint PATCH_VOXELS = PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;
#if FIXED_POINT_SCALE != 65536
#error ModelO sampling requires the Q16 lattice.
#endif
#if FEATURE_COUNT != 2
#error ModelO requires normalized HU and position, in that order.
#endif

int3 dimensions() { return int3(config[0].depth, config[0].height, config[0].width); }
int3 patchCorner() { return int3(config[0].cornerZ, config[0].cornerY, config[0].cornerX); }
bool inVolume(int3 v) { return all(v >= 0) && all(v < dimensions()); }
bool inPatch(int3 v) { return all(v >= 0) && all(v < int(PATCH_SIZE)); }
uint volumeIndex(int3 v) {
    return uint(v.x) * config[0].height * config[0].width + uint(v.y) * config[0].width + uint(v.z);
}
uint patchIndex(int3 v) {
    return uint(v.x) * PATCH_SIZE * PATCH_SIZE + uint(v.y) * PATCH_SIZE + uint(v.z);
}
int roundFixed(int value) {
    // Arithmetic shift is floor; low bits are the nonnegative Q16 fraction.
    // Avoid signed division/remainder lowering differences on negative borders.
    int quotient = value >> 16;
    int remainder = value & 65535;
    int half = FIXED_POINT_SCALE / 2;
    if (remainder < half) return quotient;
    if (remainder > half) return quotient + 1;
    return ((quotient & 1) == 0) ? quotient : quotient + 1;
}
int3 roundVoxel(int3 v) { return int3(roundFixed(v.x), roundFixed(v.y), roundFixed(v.z)); }
float3 toFloat(int3 v) { return float3(v) / float(FIXED_POINT_SCALE); }
float loadClamped(int3 v) { return volume_zyx[volumeIndex(clamp(v, 0, dimensions() - 1))]; }
float sampleHU(int3 posFP) {
    // Split before conversion: float32 whole coordinates lose low Q16 bits
    // above voxel 256, changing interpolation even though scatter stays exact.
    int3 lo = posFP >> 16;
    float3 fraction = float3(posFP & 65535) / float(FIXED_POINT_SCALE);
    float c00 = lerp(loadClamped(lo), loadClamped(lo + int3(0, 0, 1)), fraction.z);
    float c01 = lerp(loadClamped(lo + int3(0, 1, 0)), loadClamped(lo + int3(0, 1, 1)), fraction.z);
    float c10 = lerp(loadClamped(lo + int3(1, 0, 0)), loadClamped(lo + int3(1, 0, 1)), fraction.z);
    float c11 = lerp(loadClamped(lo + int3(1, 1, 0)), loadClamped(lo + int3(1, 1, 1)), fraction.z);
    float c0 = lerp(c00, c01, fraction.y);
    float c1 = lerp(c10, c11, fraction.y);
    return lerp(c0, c1, fraction.x);
}
void writeMask(uint index, bool valid) {
    uint word = index / 4;
    if (index % 4 == 0) ray_mask[word] = 0;
    ray_mask[word] |= uint(valid) << ((index % 4) * 8);
}
void writeGeometry(uint ray, int3 originFP, int3 stepFP) {
    float3 relativeBase = float3((originFP >> 16) - patchCorner());
    float3 fraction = float3(originFP & 65535) / float(FIXED_POINT_SCALE);
    float3 origin = (relativeBase + fraction) / float(PATCH_SIZE);
    float3 direction = toFloat(stepFP) / float(PATCH_SIZE);
    origin_zyx_norm[ray] = { origin.x, origin.y, origin.z };
    direction_zyx_norm[ray] = { direction.x, direction.y, direction.z };
}
void buildRay(uint ray) {
    RayGeometry geometry = ray_geometry[ray];
    int3 originFP = int3(geometry.originZ, geometry.originY, geometry.originX);
    int3 stepFP = int3(geometry.stepZ, geometry.stepY, geometry.stepX);
    float h0 = max(abs(geometry.originHU), 1.0);
    int3 corner = patchCorner();
    writeGeometry(ray, originFP, stepFP);
    uint validCount = 0;
    bool stopped = false;
    for (uint k = 0; k < RAY_LENGTH; ++k) {
        int3 posFP = originFP + stepFP * int(k + 1);
        int3 voxel = roundVoxel(posFP);
        int3 local = voxel - corner;
        // Match training: stop at the first volume or patch exit.
        bool valid = !stopped && inVolume(voxel) && inPatch(local);
        uint sample = ray * RAY_LENGTH + k;
        float hu = valid ? sampleHU(posFP) : 0.0;
        ray_hu[sample] = hu;
        ray_features[sample * FEATURE_COUNT] = hu / h0;
        voxel_indices[sample] = uint2(valid ? patchIndex(local) : 0, 0);
        writeMask(sample, valid);
        if (valid) validCount += 1;
        else stopped = true;
    }
    float lengthMinusOne = max(float(validCount) - 1.0, 1.0);
    for (uint k = 0; k < RAY_LENGTH; ++k) {
        uint sample = ray * RAY_LENGTH + k;
        ray_features[sample * FEATURE_COUNT + 1] = k < validCount ? float(k) / lengthMinusOne : 0.0;
    }
}
void buildPatch(uint index) {
    uint z = index / (PATCH_SIZE * PATCH_SIZE);
    uint rem = index - z * PATCH_SIZE * PATCH_SIZE;
    uint y = rem / PATCH_SIZE;
    uint x = rem - y * PATCH_SIZE;
    int3 voxel = patchCorner() + int3(z, y, x);
    float value = inVolume(voxel) ? volume_zyx[volumeIndex(voxel)] : 0.0;
    float h0 = config[0].h0;
    ct_patch[index] = h0 > 1.0e-6 ? value / h0 : value;
}
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint rays = config[0].rayCount;
    if (id.x < rays) { buildRay(id.x); return; }
    uint patchTask = id.x - rays;
    if (patchTask < PATCH_VOXELS) buildPatch(patchTask);
}
