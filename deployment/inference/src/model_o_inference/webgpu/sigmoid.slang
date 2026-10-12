StructuredBuffer<float> logits;
RWStructuredBuffer<float> probability;
RWStructuredBuffer<Atomic<uint>> invalid;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= PATCH_SIZE * PATCH_SIZE * PATCH_SIZE) return;
    float value = logits[id.x];
    if (!(abs(value) <= asfloat(0x7f7fffff))) invalid[0].or(1);
    probability[id.x] = 1.0 / (1.0 + exp(-value));
}
