#include "slang-cuda-prelude.h"


#line 3 "deployment/inference/src/model_o_inference/webgpu/patch_inputs.slang"
struct PatchConfig_0
{
    uint depth_0;
    uint height_0;
    uint width_0;
    uint rayCount_0;
    int cornerZ_0;
    int cornerY_0;
    int cornerX_0;
    float h0_0;
};


#line 7
struct RayGeometry_0
{
    float originHU_0;
    int originZ_0;
    int originY_0;
    int originX_0;
    int stepZ_0;
    int stepY_0;
    int stepX_0;
};


#line 13
struct PackedZYX_0
{
    float z_0;
    float y_0;
    float x_0;
};


#line 102
struct GlobalParams_0
{
    StructuredBuffer<PatchConfig_0> config_0;
    StructuredBuffer<float> volume_zyx_0;
    StructuredBuffer<RayGeometry_0> ray_geometry_0;
    RWStructuredBuffer<float> ray_features_0;
    RWStructuredBuffer<uint> ray_mask_0;
    RWStructuredBuffer<PackedZYX_0> origin_zyx_norm_0;
    RWStructuredBuffer<PackedZYX_0> direction_zyx_norm_0;
    RWStructuredBuffer<uint2 > voxel_indices_0;
    RWStructuredBuffer<float> ray_hu_0;
    RWStructuredBuffer<float> ct_patch_0;
};


#line 102
extern "C" __constant__ GlobalParams_0 SLANG_globalParams;
#define globalParams_0 (&SLANG_globalParams)

#line 13087 "hlsl.meta.slang"
__device__ int3  min_0(int3  x_1, int3  y_1)
{

#line 7378
    int3  result_0;

#line 7378
    int i_0 = int(0);

#line 7378
    for(;;)
    {

#line 7378
        if(i_0 < int(3))
        {
        }
        else
        {

#line 7378
            break;
        }

#line 7378
        *_slang_vector_get_element_ptr(&result_0, i_0) = (I32_min((_slang_vector_get_element(x_1, i_0)), (_slang_vector_get_element(y_1, i_0))));

#line 7378
        i_0 = i_0 + int(1);

#line 7378
    }

#line 7378
    return result_0;
}


#line 12825
__device__ int3  max_0(int3  x_2, int3  y_2)
{

#line 7378
    int3  result_1;

#line 7378
    int i_1 = int(0);

#line 7378
    for(;;)
    {

#line 7378
        if(i_1 < int(3))
        {
        }
        else
        {

#line 7378
            break;
        }

#line 7378
        *_slang_vector_get_element_ptr(&result_1, i_1) = (I32_max((_slang_vector_get_element(x_2, i_1)), (_slang_vector_get_element(y_2, i_1))));

#line 7378
        i_1 = i_1 + int(1);

#line 7378
    }

#line 7378
    return result_1;
}


#line 12385
__device__ float lerp_0(float x_3, float y_3, float s_0)
{

#line 12397
    return x_3 + (y_3 - x_3) * s_0;
}


#line 9106
__device__ int3  clamp_0(int3  x_4, int3  minBound_0, int3  maxBound_0)
{

#line 9124
    return min_0(max_0(x_4, minBound_0), maxBound_0);
}


#line 7701
__device__ bool all_0(bool3  x_5)
{

#line 7701
    bool result_2 = true;

#line 7701
    int i_2 = int(0);

#line 7753
    for(;;)
    {

#line 7753
        if(i_2 < int(3))
        {
        }
        else
        {

#line 7753
            break;
        }

#line 7754
        if(result_2)
        {

#line 7754
            result_2 = (bool((_slang_vector_get_element(x_5, i_2))));

#line 7754
        }
        else
        {

#line 7754
            result_2 = false;

#line 7754
        }

#line 7753
        i_2 = i_2 + int(1);

#line 7753
    }

    return result_2;
}


#line 1340 "core.meta.slang"
__device__ PatchConfig_0 slang_ldg_0(PatchConfig_0 * ptr_0)
{

#line 1340
    uint _S1 = __ldg(&ptr_0->depth_0);

#line 1340
    uint _S2 = __ldg(&ptr_0->height_0);

#line 1340
    uint _S3 = __ldg(&ptr_0->width_0);

#line 1340
    uint _S4 = __ldg(&ptr_0->rayCount_0);

#line 1340
    int _S5 = __ldg(&ptr_0->cornerZ_0);

#line 1340
    int _S6 = __ldg(&ptr_0->cornerY_0);

#line 1340
    int _S7 = __ldg(&ptr_0->cornerX_0);

#line 1340
    float _S8 = __ldg(&ptr_0->h0_0);

#line 1340
    PatchConfig_0 _S9 = { _S1, _S2, _S3, _S4, _S5, _S6, _S7, _S8 };

#line 1340
    return _S9;
}


#line 1340
__device__ RayGeometry_0 slang_ldg_1(RayGeometry_0 * ptr_1)
{

#line 1340
    float _S10 = __ldg(&ptr_1->originHU_0);

#line 1340
    int _S11 = __ldg(&ptr_1->originZ_0);

#line 1340
    int _S12 = __ldg(&ptr_1->originY_0);

#line 1340
    int _S13 = __ldg(&ptr_1->originX_0);

#line 1340
    int _S14 = __ldg(&ptr_1->stepZ_0);

#line 1340
    int _S15 = __ldg(&ptr_1->stepY_0);

#line 1340
    int _S16 = __ldg(&ptr_1->stepX_0);

#line 1340
    RayGeometry_0 _S17 = { _S10, _S11, _S12, _S13, _S14, _S15, _S16 };

#line 1340
    return _S17;
}


#line 35 "deployment/inference/src/model_o_inference/webgpu/patch_inputs.slang"
__device__ int3  patchCorner_0()
{

#line 35
    PatchConfig_0 _S18 = slang_ldg_0((&(globalParams_0->config_0)[int(0)]));

#line 35
    PatchConfig_0 _S19 = slang_ldg_0((&(globalParams_0->config_0)[int(0)]));

#line 35
    PatchConfig_0 _S20 = slang_ldg_0((&(globalParams_0->config_0)[int(0)]));

#line 35
    return make_int3 (_S18.cornerZ_0, _S19.cornerY_0, _S20.cornerX_0);
}


#line 55
__device__ float3  toFloat_0(int3  v_0)
{

#line 55
    float3  _S21 = make_float3 ((float)v_0.x, (float)v_0.y, (float)v_0.z);

#line 55
    return _S21 / make_float3 (6.5536e+04f);
}


#line 13
__device__ PackedZYX_0 PackedZYX_x24init_0(float z_1, float y_4, float x_6)
{

#line 13
    PackedZYX_0 _S22;

#line 13
    (&_S22)->z_0 = z_1;

#line 13
    (&_S22)->y_0 = y_4;

#line 13
    (&_S22)->x_0 = x_6;

#line 13
    return _S22;
}


#line 75
__device__ void writeGeometry_0(uint ray_0, int3  originFP_0, int3  stepFP_0)
{

#line 76
    int3  _S23 = (originFP_0 >> make_int3 (int(16))) - patchCorner_0();

#line 76
    float3  _S24 = make_float3 ((float)_S23.x, (float)_S23.y, (float)_S23.z);
    int3  _S25 = originFP_0 & make_int3 (int(65535));

#line 77
    float3  _S26 = make_float3 ((float)_S25.x, (float)_S25.y, (float)_S25.z);
    float3  origin_0 = (_S24 + _S26 / make_float3 (6.5536e+04f)) / make_float3 (48.0f);
    float3  direction_0 = toFloat_0(stepFP_0) / make_float3 (48.0f);
    *(&(globalParams_0->origin_zyx_norm_0)[ray_0]) = PackedZYX_x24init_0(origin_0.x, origin_0.y, origin_0.z);
    *(&(globalParams_0->direction_zyx_norm_0)[ray_0]) = PackedZYX_x24init_0(direction_0.x, direction_0.y, direction_0.z);
    return;
}


#line 44
__device__ int roundFixed_0(int value_0)
{

    int quotient_0 = value_0 >> int(16);
    int remainder_0 = value_0 & int(65535);

    if(remainder_0 < int(32768))
    {

#line 50
        return quotient_0;
    }

#line 51
    if(remainder_0 > int(32768))
    {

#line 51
        return quotient_0 + int(1);
    }

#line 51
    int _S27;
    if((quotient_0 & int(1)) == int(0))
    {

#line 52
        _S27 = quotient_0;

#line 52
    }
    else
    {

#line 52
        _S27 = quotient_0 + int(1);

#line 52
    }

#line 52
    return _S27;
}


#line 54
__device__ int3  roundVoxel_0(int3  v_1)
{

#line 54
    return make_int3 (roundFixed_0(v_1.x), roundFixed_0(v_1.y), roundFixed_0(v_1.z));
}


#line 34
__device__ int3  dimensions_0()
{

#line 34
    PatchConfig_0 _S28 = slang_ldg_0((&(globalParams_0->config_0)[int(0)]));

#line 34
    int _S29 = int(_S28.depth_0);

#line 34
    PatchConfig_0 _S30 = slang_ldg_0((&(globalParams_0->config_0)[int(0)]));

#line 34
    int _S31 = int(_S30.height_0);

#line 34
    PatchConfig_0 _S32 = slang_ldg_0((&(globalParams_0->config_0)[int(0)]));

#line 34
    return make_int3 (_S29, _S31, int(_S32.width_0));
}


#line 36
__device__ bool inVolume_0(int3  v_2)
{

#line 36
    bool _S33;

#line 36
    if(all_0(v_2 >= make_int3 (int(0))))
    {

#line 36
        _S33 = all_0(v_2 < (dimensions_0()));

#line 36
    }
    else
    {

#line 36
        _S33 = false;

#line 36
    }

#line 36
    return _S33;
}


#line 37
__device__ bool inPatch_0(int3  v_3)
{

#line 37
    bool _S34;

#line 37
    if(all_0(v_3 >= make_int3 (int(0))))
    {

#line 37
        _S34 = all_0(v_3 < make_int3 (int(48)));

#line 37
    }
    else
    {

#line 37
        _S34 = false;

#line 37
    }

#line 37
    return _S34;
}


#line 38
__device__ uint volumeIndex_0(int3  v_4)
{

#line 39
    uint _S35 = uint(v_4.x);

#line 39
    PatchConfig_0 _S36 = slang_ldg_0((&(globalParams_0->config_0)[int(0)]));

#line 39
    uint _S37 = _S35 * _S36.height_0;

#line 39
    PatchConfig_0 _S38 = slang_ldg_0((&(globalParams_0->config_0)[int(0)]));

#line 39
    uint _S39 = _S37 * _S38.width_0;

#line 39
    uint _S40 = uint(v_4.y);

#line 39
    PatchConfig_0 _S41 = slang_ldg_0((&(globalParams_0->config_0)[int(0)]));

#line 39
    return _S39 + _S40 * _S41.width_0 + uint(v_4.z);
}


#line 56
__device__ float loadClamped_0(int3  v_5)
{

#line 56
    float _S42 = __ldg((&(globalParams_0->volume_zyx_0)[volumeIndex_0(clamp_0(v_5, make_int3 (int(0)), dimensions_0() - make_int3 (int(1))))]));

#line 56
    return _S42;
}


#line 57
__device__ float sampleHU_0(int3  posFP_0)
{

    int3  lo_0 = posFP_0 >> make_int3 (int(16));
    int3  _S43 = posFP_0 & make_int3 (int(65535));

#line 61
    float3  _S44 = make_float3 ((float)_S43.x, (float)_S43.y, (float)_S43.z);

#line 61
    float3  fraction_0 = _S44 / make_float3 (6.5536e+04f);
    float _S45 = fraction_0.z;



    float _S46 = fraction_0.y;

    return lerp_0(lerp_0(lerp_0(loadClamped_0(lo_0), loadClamped_0(lo_0 + make_int3 (int(0), int(0), int(1))), _S45), lerp_0(loadClamped_0(lo_0 + make_int3 (int(0), int(1), int(0))), loadClamped_0(lo_0 + make_int3 (int(0), int(1), int(1))), _S45), _S46), lerp_0(lerp_0(loadClamped_0(lo_0 + make_int3 (int(1), int(0), int(0))), loadClamped_0(lo_0 + make_int3 (int(1), int(0), int(1))), _S45), lerp_0(loadClamped_0(lo_0 + make_int3 (int(1), int(1), int(0))), loadClamped_0(lo_0 + make_int3 (int(1), int(1), int(1))), _S45), _S46), fraction_0.x);
}


#line 41
__device__ uint patchIndex_0(int3  v_6)
{

#line 42
    return uint(v_6.x) * 48U * 48U + uint(v_6.y) * 48U + uint(v_6.z);
}


#line 70
__device__ void writeMask_0(uint index_0, bool valid_0)
{

#line 71
    uint word_0 = index_0 / 4U;
    uint _S47 = index_0 % 4U;

#line 72
    if(_S47 == 0U)
    {

#line 72
        *(&(globalParams_0->ray_mask_0)[word_0]) = 0U;

#line 72
    }
    uint * _S48 = (&(globalParams_0->ray_mask_0)[word_0]);

#line 73
    *_S48 = (*_S48) | (uint(valid_0) << (_S47 * 8U));
    return;
}


#line 83
__device__ void buildRay_0(uint ray_1)
{

#line 83
    float hu_0;

#line 83
    RayGeometry_0 _S49 = slang_ldg_1((&(globalParams_0->ray_geometry_0)[ray_1]));

    int3  originFP_1 = make_int3 (_S49.originZ_0, _S49.originY_0, _S49.originX_0);
    int3  stepFP_1 = make_int3 (_S49.stepZ_0, _S49.stepY_0, _S49.stepX_0);
    float _S50 = (F32_max(((F32_abs((_S49.originHU_0)))), (1.0f)));
    int3  _S51 = patchCorner_0();
    writeGeometry_0(ray_1, originFP_1, stepFP_1);

#line 89
    bool stopped_0 = false;

#line 89
    uint k_0 = 0U;

#line 89
    uint validCount_0 = 0U;


    for(;;)
    {

#line 92
        if(k_0 < 120U)
        {
        }
        else
        {

#line 92
            break;
        }

#line 93
        uint _S52 = k_0 + 1U;

#line 93
        int3  posFP_1 = originFP_1 + stepFP_1 * make_int3 (int(_S52));
        int3  voxel_0 = roundVoxel_0(posFP_1);
        int3  local_0 = voxel_0 - _S51;

#line 95
        bool _S53;

        if(!stopped_0)
        {

#line 97
            _S53 = inVolume_0(voxel_0);

#line 97
        }
        else
        {

#line 97
            _S53 = false;

#line 97
        }

#line 97
        bool valid_1;

#line 97
        if(_S53)
        {

#line 97
            valid_1 = inPatch_0(local_0);

#line 97
        }
        else
        {

#line 97
            valid_1 = false;

#line 97
        }
        uint sample_0 = ray_1 * 120U + k_0;
        if(valid_1)
        {

#line 99
            hu_0 = sampleHU_0(posFP_1);

#line 99
        }
        else
        {

#line 99
            hu_0 = 0.0f;

#line 99
        }
        *(&(globalParams_0->ray_hu_0)[sample_0]) = hu_0;
        *(&(globalParams_0->ray_features_0)[sample_0 * 2U]) = hu_0 / _S50;
        uint2  * _S54 = (&(globalParams_0->voxel_indices_0)[sample_0]);

#line 102
        uint _S55;

#line 102
        if(valid_1)
        {

#line 102
            _S55 = patchIndex_0(local_0);

#line 102
        }
        else
        {

#line 102
            _S55 = 0U;

#line 102
        }

#line 102
        *_S54 = make_uint2 (_S55, 0U);
        writeMask_0(sample_0, valid_1);
        if(valid_1)
        {

#line 104
            validCount_0 = validCount_0 + 1U;

#line 104
        }
        else
        {

#line 104
            stopped_0 = true;

#line 104
        }

#line 92
        k_0 = _S52;

#line 92
    }

#line 107
    float _S56 = (F32_max((float(validCount_0) - 1.0f), (1.0f)));

#line 107
    k_0 = 0U;
    for(;;)
    {

#line 108
        if(k_0 < 120U)
        {
        }
        else
        {

#line 108
            break;
        }
        float * _S57 = (&(globalParams_0->ray_features_0)[(ray_1 * 120U + k_0) * 2U + 1U]);

#line 110
        if(k_0 < validCount_0)
        {

#line 110
            hu_0 = float(k_0) / _S56;

#line 110
        }
        else
        {

#line 110
            hu_0 = 0.0f;

#line 110
        }

#line 110
        *_S57 = hu_0;

#line 108
        k_0 = k_0 + 1U;

#line 108
    }



    return;
}


#line 113
__device__ void buildPatch_0(uint index_1)
{

#line 114
    uint z_2 = index_1 / 2304U;
    uint rem_0 = index_1 - z_2 * 48U * 48U;
    uint y_5 = rem_0 / 48U;

    int3  voxel_1 = patchCorner_0() + make_int3 (int(z_2), int(y_5), int(rem_0 - y_5 * 48U));

#line 118
    float value_1;
    if(inVolume_0(voxel_1))
    {

#line 119
        float _S58 = __ldg((&(globalParams_0->volume_zyx_0)[volumeIndex_0(voxel_1)]));

#line 119
        value_1 = _S58;

#line 119
    }
    else
    {

#line 119
        value_1 = 0.0f;

#line 119
    }

#line 119
    PatchConfig_0 _S59 = slang_ldg_0((&(globalParams_0->config_0)[int(0)]));

    float * _S60 = (&(globalParams_0->ct_patch_0)[index_1]);

#line 121
    if((_S59.h0_0) > 9.99999997475242708e-07f)
    {

#line 121
        value_1 = value_1 / _S59.h0_0;

#line 121
    }

#line 121
    *_S60 = value_1;
    return;
}

extern "C" __global__ void main_0()
{

#line 125
    uint3  _S61 = blockIdx * blockDim + threadIdx;

#line 125
    PatchConfig_0 _S62 = slang_ldg_0((&(globalParams_0->config_0)[int(0)]));

    uint _S63 = _S61.x;

#line 127
    if(_S63 < (_S62.rayCount_0))
    {

#line 127
        buildRay_0(_S63);

#line 127
        return;
    }

#line 128
    uint patchTask_0 = _S63 - _S62.rayCount_0;
    if(patchTask_0 < 110592U)
    {

#line 129
        buildPatch_0(patchTask_0);

#line 129
    }
    return;
}
