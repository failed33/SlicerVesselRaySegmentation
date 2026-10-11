"""Shaders distributed with the ModelO runtime."""

from pathlib import Path

# TODO: replace with `SlangPy`

SHADER_PATH = Path(__file__).with_name("webgpu") / "model_o_patch_inputs.wgsl"

SIGMOID = """
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> probability: array<f32>;
@group(0) @binding(2) var<storage, read_write> invalid: atomic<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < 48u * 48u * 48u) {
        if (!(abs(logits[id.x]) <= bitcast<f32>(0x7f7fffffu))) {
            atomicOr(&invalid, 1u);
        }
        probability[id.x] = 1.0 / (1.0 + exp(-logits[id.x]));
    }
}
"""
