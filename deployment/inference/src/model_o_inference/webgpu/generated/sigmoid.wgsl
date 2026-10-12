@binding(0) @group(0) var<storage, read> logits_0 : array<f32>;

@binding(2) @group(0) var<storage, read_write> invalid_0 : array<atomic<u32>>;

@binding(1) @group(0) var<storage, read_write> probability_0 : array<f32>;

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id_0 : vec3<u32>)
{
    var _S1 : u32 = id_0.x;
    if(_S1 >= u32(110592))
    {
        return;
    }
    var value_0 : f32 = logits_0[_S1];
    if(!((abs(value_0)) <= (bitcast<f32>((i32(2139095039))))))
    {
        var _S2 : u32 = atomicOr(&(invalid_0[i32(0)]), u32(1));
    }
    probability_0[_S1] = 1.0f / (1.0f + exp(- value_0));
    return;
}

