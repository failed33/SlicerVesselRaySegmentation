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
static const float NEG_INF = -1.0e30;

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
    int quotient = value / FIXED_POINT_SCALE;
    int remainder = value % FIXED_POINT_SCALE;
    if (remainder < 0) { quotient -= 1; remainder += FIXED_POINT_SCALE; }
    int half = FIXED_POINT_SCALE / 2;
    if (remainder < half) return quotient;
    if (remainder > half) return quotient + 1;
    return (quotient % 2 == 0) ? quotient : quotient + 1;
}
int3 roundVoxel(int3 v) { return int3(roundFixed(v.x), roundFixed(v.y), roundFixed(v.z)); }
float3 toFloat(int3 v) { return float3(v) / float(FIXED_POINT_SCALE); }
float loadClamped(int3 v) { return volume_zyx[volumeIndex(clamp(v, 0, dimensions() - 1))]; }
float sampleHU(float3 pos) {
    int3 lo = int3(floor(pos));
    float3 fraction = pos - float3(lo);
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
uint2 paddingIndex() {
    // Existing CPU padding represents absolute voxel (0,0,0), with mask false.
    int3 local = -patchCorner();
    return uint2(inPatch(local) ? patchIndex(local) : 0, 0);
}
void writeGeometry(uint ray, int3 originFP, int3 stepFP) {
    float3 origin = (toFloat(originFP) - float3(patchCorner())) / float(PATCH_SIZE);
    float3 direction = toFloat(stepFP) / float(PATCH_SIZE);
    origin_zyx_norm[ray] = { origin.x, origin.y, origin.z };
    direction_zyx_norm[ray] = { direction.x, direction.y, direction.z };
}
void zeroFeatures(uint sample) {
    for (uint feature = 0; feature < FEATURE_COUNT; ++feature)
        ray_features[sample * FEATURE_COUNT + feature] = 0.0;
}
void buildRay(uint ray) {
    float huValues[RAY_LENGTH];
    uint validValues[RAY_LENGTH];
    RayGeometry geometry = ray_geometry[ray];
    int3 originFP = int3(geometry.originZ, geometry.originY, geometry.originX);
    int3 stepFP = int3(geometry.stepZ, geometry.stepY, geometry.stepX);
    float h0 = max(abs(geometry.originHU), 1.0);
    int3 corner = patchCorner();
    writeGeometry(ray, originFP, stepFP);
    uint validCount = 0;
    float totalSum = 0.0;
    bool stopped = false;
    for (uint k = 0; k < RAY_LENGTH; ++k) {
        int3 posFP = originFP + stepFP * int(k + 1);
        int3 voxel = roundVoxel(posFP);
        bool valid = !stopped && inVolume(voxel);
        uint sample = ray * RAY_LENGTH + k;
        if (valid) {
            float hu = sampleHU(toFloat(posFP));
            int3 local = voxel - corner;
            bool scatter = inPatch(local);
            huValues[k] = hu;
            validValues[k] = 1;
            ray_hu[sample] = hu;
            voxel_indices[sample] = uint2(scatter ? patchIndex(local) : 0, 0);
            writeMask(sample, scatter);
            validCount += 1;
            totalSum += hu;
        } else {
            stopped = true;
            huValues[k] = 0.0;
            validValues[k] = 0;
            ray_hu[sample] = 0.0;
            voxel_indices[sample] = paddingIndex();
            writeMask(sample, false);
        }
    }
    if (validCount == 0) {
        for (uint k = 0; k < RAY_LENGTH; ++k) zeroFeatures(ray * RAY_LENGTH + k);
        return;
    }
    float n = float(validCount);
    float mean = totalSum / n;
    float varianceSum = 0.0;
    for (uint k = 0; k < RAY_LENGTH; ++k) {
        if (validValues[k] != 0) {
            float delta = huValues[k] - mean;
            varianceSum += delta * delta;
        }
    }
    float variance = varianceSum / n;
    float prefixSum = 0.0;
    float runningMax = NEG_INF;
    float previousHU = 0.0;
    float lengthMinusOne = max(n - 1.0, 1.0);
    for (uint k = 0; k < RAY_LENGTH; ++k) {
        uint sample = ray * RAY_LENGTH + k;
        if (validValues[k] == 0) { zeroFeatures(sample); continue; }
        float hu = huValues[k];
        prefixSum += hu;
        runningMax = max(runningMax, hu);
        float count = float(k + 1);
        float suffixSum = totalSum - prefixSum;
        float suffixCount = max(n - count, 1.0);
        float muL = prefixSum / count;
        float muR = suffixSum / suffixCount;
        float rawEta = count * max(n - count, 0.0) * (muL - muR) * (muL - muR);
        float eta2 = variance > 1.0e-12 ? min(rawEta / (n * variance), 1.0) : 0.0;
        float gradient = k > 0 ? (hu - previousHU) / h0 : 0.0;
        uint base = sample * FEATURE_COUNT;
        ray_features[base] = hu / h0;
        ray_features[base + 1] = gradient;
        ray_features[base + 2] = (prefixSum / count) / h0;
        ray_features[base + 3] = (muL - muR) / h0;
        ray_features[base + 4] = eta2;
        ray_features[base + 5] = (runningMax - hu) / h0;
        ray_features[base + 6] = float(k) / lengthMinusOne;
        previousHU = hu;
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
