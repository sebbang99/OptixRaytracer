/*

 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optix.h>
#include <vector_types.h>

#include "helpers.h"
#include "whitted.h"

#define TOON_SHADING 1

// Sehee added begin
__device__ float radians(float a)
{
    return 0.017453292f * a;
}
// Sehee added end

extern "C" {
__constant__ whitted::LaunchParams params;
}

__device__ float3 make_float_3(const float *x) {
    float3 f3;
    f3.x = x[0]; f3.y = x[1]; f3.z = x[2];
    return f3;
}

static __device__ __inline__ whitted::PayloadRadiance getPayloadRadiance()
{
    whitted::PayloadRadiance prd;
    prd.result.x   = __uint_as_float( optixGetPayload_0() );
    prd.result.y   = __uint_as_float( optixGetPayload_1() );
    prd.result.z   = __uint_as_float( optixGetPayload_2() );
    prd.importance = __uint_as_float( optixGetPayload_3() );
    prd.depth      = optixGetPayload_4();
    return prd;
}

static __device__ __inline__ void setPayloadRadiance( const whitted::PayloadRadiance& prd )
{
    optixSetPayload_0( __float_as_uint( prd.result.x ) );
    optixSetPayload_1( __float_as_uint( prd.result.y ) );
    optixSetPayload_2( __float_as_uint( prd.result.z ) );
    optixSetPayload_3( __float_as_uint( prd.importance ) );
    optixSetPayload_4( prd.depth );
}

static __device__ __inline__ whitted::PayloadOcclusion getPayloadOcclusion()
{
    whitted::PayloadOcclusion prd;
    prd.result.x = __uint_as_float( optixGetPayload_0() );
    prd.result.y = __uint_as_float( optixGetPayload_1() );
    prd.result.z = __uint_as_float( optixGetPayload_2() );
    return prd;
}

static __device__ __inline__ void setPayloadOcclusion( const whitted::PayloadOcclusion& prd )
{
    optixSetPayload_0( __float_as_uint( prd.result.x ) );
    optixSetPayload_1( __float_as_uint( prd.result.y ) );
    optixSetPayload_2( __float_as_uint( prd.result.z ) );
}

static __device__ __inline__ float3 traceRadianceRay( float3 origin, float3 direction, int depth, float importance )
{
    whitted::PayloadRadiance prd;
    prd.depth      = depth;
    prd.importance = importance;

    optixTrace( params.handle, origin, direction, params.scene_epsilon, 1e16f, 0.0f, OptixVisibilityMask( 1 ), OPTIX_RAY_FLAG_NONE,
                whitted::RAY_TYPE_RADIANCE, whitted::RAY_TYPE_COUNT, whitted::RAY_TYPE_RADIANCE, float3_as_args( prd.result ),
                /* Can't use __float_as_uint() because it returns rvalue but payload requires a lvalue */
                reinterpret_cast<unsigned int&>( prd.importance ), reinterpret_cast<unsigned int&>( prd.depth ) );

    return prd.result;
}

static __device__ void phongShadowed()
{
    // this material is opaque, so it fully attenuates all shadow rays
    whitted::PayloadOcclusion prd;
    prd.result = make_float3( 0.f );
    setPayloadOcclusion( prd );
}

static __device__ void phongShade( float3 p_Kd, float3 p_Ka, float3 p_Ks, float3 p_Kr, float p_phong_exp, float3 p_normal )
{
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_t    = optixGetRayTmax();

    whitted::PayloadRadiance prd = getPayloadRadiance();

    float3 hit_point = ray_orig + ray_t * ray_dir;

    // ambient contribution
    Light::Ambient ambient_light = params.lights[0].ambient;
    float3         result        = p_Ka * ambient_light.color;

    // compute direct lighting
    Light::Point point_light = params.lights[1].point;
    float        Ldist       = length( point_light.position - hit_point );
    float3       L           = normalize( point_light.position - hit_point );
    float        nDl         = dot( p_normal, L );

    // cast shadow ray
    float3 light_attenuation = make_float3( static_cast<float>( nDl > 0.0f ) );
    if( nDl > 0.0f )
    {
        whitted::PayloadOcclusion shadow_prd;
        shadow_prd.result = make_float3( 1.0f );

        optixTrace( params.handle, hit_point, L, 0.01f, Ldist, 0.0f, OptixVisibilityMask( 1 ), OPTIX_RAY_FLAG_NONE,
                    whitted::RAY_TYPE_OCCLUSION, whitted::RAY_TYPE_COUNT, whitted::RAY_TYPE_OCCLUSION,
                    float3_as_args( shadow_prd.result ) );

        light_attenuation = shadow_prd.result;
    }

    // If not completely shadowed, light the hit point
    if( fmaxf( light_attenuation ) > 0.0f )
    {
        float3 Lc = point_light.color * light_attenuation;

        result += p_Kd * nDl * Lc;

        float3 H   = normalize( L - ray_dir );
        float  nDh = dot( p_normal, H );
        if( nDh > 0 )
        {
            float power = pow( nDh, p_phong_exp );
            result += p_Ks * power * Lc;
        }
    }

    if( fmaxf( p_Kr ) > 0 )
    {

        // ray tree attenuation
        float new_importance = prd.importance * luminance( p_Kr );
        int   new_depth      = prd.depth + 1;

        // reflection ray
        // compare new_depth to max_depth - 1 to leave room for a potential shadow ray trace
        if( new_importance >= 0.01f && new_depth <= params.max_depth - 1 )
        {
            float3 R = reflect( ray_dir, p_normal );

            result += p_Kr * traceRadianceRay( hit_point, R, new_depth, new_importance );
        }
    }

    // compute spot lighting
    const uint32_t spot_light_cnt = 2;
    const uint32_t spot_light_begin = 2;
    for (uint32_t i = 0; i < spot_light_cnt; i++) {
        Light::Spot spot_light = params.lights[spot_light_begin + i].spot;
        float spot_Ldist = length(spot_light.position - hit_point);
        float3 spot_L = normalize(spot_light.position - hit_point);
        float spot_nDl = dot(p_normal, spot_L);

        float3 spot_result_color = make_float3(0.0f, 0.0f, 0.0f);
        float spot_factor = dot(-spot_L, spot_light.direction);
        if (spot_factor > cos(radians(spot_light.cutoff))) {
            spot_result_color = pow(spot_factor, 2.0f) * make_float3(1.0f, 1.0f, 1.0f);

            float3 spot_light_attenuation = make_float3(static_cast<float>(spot_nDl > 0.0f));
            if (spot_nDl > 0.0f)
            {
                whitted::PayloadOcclusion shadow_prd;
                shadow_prd.result = make_float3(1.0f);

                optixTrace(params.handle, hit_point, spot_L, 0.01f, spot_Ldist, 0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
                    whitted::RAY_TYPE_OCCLUSION, whitted::RAY_TYPE_COUNT, whitted::RAY_TYPE_OCCLUSION,
                    float3_as_args(shadow_prd.result));

                spot_light_attenuation = shadow_prd.result;
            }

            if (fmaxf(spot_light_attenuation) > 0.0f)
            {
                float3 spot_Lc = spot_light.color * spot_light_attenuation;

                result += p_Kd * spot_nDl * spot_Lc * spot_result_color;

                float3 spot_H = normalize(spot_L - ray_dir);
                float  spot_nDh = dot(p_normal, spot_H);
                if (spot_nDh > 0)
                {
                    float spot_power = pow(spot_nDh, p_phong_exp);
                    result += p_Ks * spot_power * spot_Lc * spot_result_color;
                }
            }
        }
    }

    // pass the color back
    prd.result = result;
    setPayloadRadiance( prd );
}

static __device__ void phongToonShade(float3 p_Kd, float3 p_Ka, float3 p_Ks, float3 p_Kr, float p_phong_exp, float3 p_normal)
{
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float  ray_t = optixGetRayTmax();

    whitted::PayloadRadiance prd = getPayloadRadiance();

    float3 hit_point = ray_orig + ray_t * ray_dir;

    // ambient contribution
    Light::Ambient ambient_light = params.lights[0].ambient;
    float3         result = p_Ka * ambient_light.color;

    // compute direct lighting
    Light::Point point_light = params.lights[1].point;
    float        Ldist = length(point_light.position - hit_point);
    float3       L = normalize(point_light.position - hit_point);
    float        nDl = dot(p_normal, L);

    // cast shadow ray
    float3 light_attenuation = make_float3(static_cast<float>(nDl > 0.0f));
    if (nDl > 0.0f)
    {
        whitted::PayloadOcclusion shadow_prd;
        shadow_prd.result = make_float3(1.0f);

        optixTrace(params.handle, hit_point, L, 0.01f, Ldist, 0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
            whitted::RAY_TYPE_OCCLUSION, whitted::RAY_TYPE_COUNT, whitted::RAY_TYPE_OCCLUSION,
            float3_as_args(shadow_prd.result));

        light_attenuation = shadow_prd.result;
    }

    // If not completely shadowed, light the hit point
    if (fmaxf(light_attenuation) > 0.0f)
    {
        float toon_diffuse = fmaxf(0.0f, fminf(1.0f, nDl));
        toon_diffuse = (toon_diffuse < 0.5f ? 0.3f : 1.0f);

        float3 Lc = point_light.color * light_attenuation;

        result += p_Kd * toon_diffuse * Lc;

        float3 H = normalize(L - ray_dir);
        float  nDh = dot(p_normal, H);
        if (nDh > 0)
        {
            float power = pow(nDh, p_phong_exp);
            float toon_specular = (power > 0.5f ? 1.0f : 0.0f);

            // strange rim lighting ...
            //float rim_value = 1.0f - dot(ray_dir, p_normal);
            //float3 rim_dot = make_float3(rim_value, rim_value, rim_value);
            result += p_Ks * (toon_specular * Lc/* + rim_dot * 0.1f*/);
        }
    }

    if (fmaxf(p_Kr) > 0)
    {

        // ray tree attenuation
        float new_importance = prd.importance * luminance(p_Kr);
        int   new_depth = prd.depth + 1;

        // reflection ray
        // compare new_depth to max_depth - 1 to leave room for a potential shadow ray trace
        if (new_importance >= 0.01f && new_depth <= params.max_depth - 1)
        {
            float3 R = reflect(ray_dir, p_normal);

            result += p_Kr * traceRadianceRay(hit_point, R, new_depth, new_importance);
        }
    }

    // compute spot lighting
    Light::Spot spot_light = params.lights[2].spot;
    float spot_Ldist = length(spot_light.position - hit_point);
    float3 spot_L = normalize(spot_light.position - hit_point);
    float spot_nDl = dot(p_normal, spot_L);

    float3 spot_result_color = make_float3(0.0f, 0.0f, 0.0f);
    float spot_factor = dot(-spot_L, spot_light.direction);
    if (spot_factor > cos(radians(spot_light.cutoff))) {
        spot_result_color = pow(spot_factor, 2.0f) * make_float3(1.0f, 1.0f, 1.0f);

        float3 spot_light_attenuation = make_float3(static_cast<float>(spot_nDl > 0.0f));
        if (spot_nDl > 0.0f)
        {
            whitted::PayloadOcclusion shadow_prd;
            shadow_prd.result = make_float3(1.0f);

            optixTrace(params.handle, hit_point, spot_L, 0.01f, spot_Ldist, 0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
                whitted::RAY_TYPE_OCCLUSION, whitted::RAY_TYPE_COUNT, whitted::RAY_TYPE_OCCLUSION,
                float3_as_args(shadow_prd.result));

            spot_light_attenuation = shadow_prd.result;
        }

        if (fmaxf(spot_light_attenuation) > 0.0f)
        {
            float toon_spot_diffuse = fmaxf(0.0f, fminf(1.0f, nDl));
            toon_spot_diffuse = (toon_spot_diffuse < 0.5f ? 0.3f : 1.0f);

            float3 spot_Lc = spot_light.color * spot_light_attenuation;

            result += p_Kd * toon_spot_diffuse * spot_Lc * spot_result_color;

            float3 spot_H = normalize(spot_L - ray_dir);
            float  spot_nDh = dot(p_normal, spot_H);
            if (spot_nDh > 0)
            {
                float spot_power = pow(spot_nDh, p_phong_exp);
                float toon_specular = (spot_power > 0.5f ? 1.0f : 0.0f);
                result += p_Ks * toon_specular * spot_Lc * spot_result_color;
            }
        }
    }

    // just for debugging
    //result = make_float3(0.0f, 1.0f, 0.0f);

    // pass the color back
    prd.result = result;
    setPayloadRadiance(prd);
}

static __device__ void phongShadePointCloud(float3 p_Kd, float3 p_Ka, float3 p_Ks, float3 p_Kr, float p_phong_exp, float3 p_normal)
{
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float  ray_t = optixGetRayTmax();

    whitted::PayloadRadiance prd = getPayloadRadiance();

    float3 hit_point = ray_orig + ray_t * ray_dir;

    // ambient contribution
    Light::Ambient ambient_light = params.lights[0].ambient;
    float3         result = p_Ka * ambient_light.color;

    // compute direct lighting
    Light::Point point_light = params.lights[1].point;
    float        Ldist = length(point_light.position - hit_point);
    float3       L = normalize(point_light.position - hit_point);
    float        nDl = dot(p_normal, L);

    float3 light_attenuation = make_float3(static_cast<float>(nDl > 0.0f));

    // If not completely shadowed, light the hit point
    if (fmaxf(light_attenuation) > 0.0f)
    {
        float3 Lc = point_light.color * light_attenuation;

        result += p_Kd * nDl * Lc;

        float3 H = normalize(L - ray_dir);
        float  nDh = dot(p_normal, H);
        if (nDh > 0)
        {
            float power = pow(nDh, p_phong_exp);
            result += p_Ks * power * Lc;
        }
    }

    if (fmaxf(p_Kr) > 0)
    {

        // ray tree attenuation
        float new_importance = prd.importance * luminance(p_Kr);
        int   new_depth = prd.depth + 1;

        // reflection ray
        // compare new_depth to max_depth - 1 to leave room for a potential shadow ray trace
        if (new_importance >= 0.01f && new_depth <= params.max_depth - 1)
        {
            float3 R = reflect(ray_dir, p_normal);

            result += p_Kr * traceRadianceRay(hit_point, R, new_depth, new_importance);
        }
    }

    // compute spot lighting
    Light::Spot spot_light = params.lights[2].spot;
    float spot_Ldist = length(spot_light.position - hit_point);
    float3 spot_L = normalize(spot_light.position - hit_point);
    float spot_nDl = dot(p_normal, spot_L);

    float3 spot_result_color = make_float3(0.0f, 0.0f, 0.0f);
    float spot_factor = dot(-spot_L, spot_light.direction);
    if (spot_factor > cos(radians(spot_light.cutoff))) {
        spot_result_color = pow(spot_factor, 2.0f) * make_float3(1.0f, 1.0f, 1.0f);

        float3 spot_light_attenuation = make_float3(static_cast<float>(spot_nDl > 0.0f));

        if (fmaxf(spot_light_attenuation) > 0.0f)
        {
            float3 spot_Lc = spot_light.color * spot_light_attenuation;

            result += p_Kd * spot_nDl * spot_Lc * spot_result_color;

            float3 spot_H = normalize(spot_L - ray_dir);
            float  spot_nDh = dot(p_normal, spot_H);
            if (spot_nDh > 0)
            {
                float spot_power = pow(spot_nDh, p_phong_exp);
                result += p_Ks * spot_power * spot_Lc * spot_result_color;
            }
        }
    }

    // pass the color back
    prd.result = result;
    setPayloadRadiance(prd);
}

extern "C" __global__ void __closesthit__aabb_radiance()
{
    const whitted::HitGroupData* sbt_data = (whitted::HitGroupData*)optixGetSbtDataPointer();
    const MaterialData::Phong& phong = sbt_data->material_data.pink_mirror;

    float3 object_normal = make_float3(__uint_as_float(optixGetAttribute_0()), __uint_as_float(optixGetAttribute_1()),
        __uint_as_float(optixGetAttribute_2()));

    float3 world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(object_normal));
    float3 ffnormal = faceforward(world_normal, -optixGetWorldRayDirection(), world_normal);
    phongShade(phong.Kd, phong.Ka, phong.Ks, phong.Kr, phong.phong_exp, ffnormal);
}

extern "C" __global__ void __closesthit__mesh_radiance()
{
    // ver.1 (MyTriangleMesh)
    //const whitted::HitGroupData* sbt_data = (whitted::HitGroupData*)optixGetSbtDataPointer();
    //const MaterialData::Phong& phong = sbt_data->material_data.metal;

    //const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    //const unsigned int           gasSbtIdx = optixGetSbtGASIndex();
    //const unsigned int           primIdx = optixGetPrimitiveIndex();
    //float3 vertices[3] = {};
    //optixGetTriangleVertexData(
    //    gas,
    //    primIdx,
    //    gasSbtIdx,
    //    0,
    //    vertices);

    //float3 object_normal = cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);

    //float3 world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(object_normal));
    //float3 ffnormal = faceforward(world_normal, -optixGetWorldRayDirection(), world_normal);
    //phongShade(phong.Kd, phong.Ka, phong.Ks, phong.Kr, phong.phong_exp, ffnormal);

    // ver.2 (TriangleMesh)
    const whitted::HitGroupData* sbt_data = (whitted::HitGroupData*)optixGetSbtDataPointer();
    const MaterialData::Phong& phong = sbt_data->material_data.red_velvet;
    const GeometryData::TriangleMesh& mesh = sbt_data->geometry_data.getTriangleMesh();

    //float2 barycentrics = make_float2(__uint_as_float(optixGetAttribute_0()), __uint_as_float(optixGetAttribute_1()));
    float2 barycentrics = optixGetTriangleBarycentrics();

    const unsigned int primitiveIndex = optixGetPrimitiveIndex() * 3;

    const unsigned int* index_buffer = reinterpret_cast<const unsigned int*>(mesh.indices.data);
    unsigned int i0 = index_buffer[primitiveIndex];
    unsigned int i1 = index_buffer[primitiveIndex + 1];
    unsigned int i2 = index_buffer[primitiveIndex + 2];

    const float3* normal_buffer = reinterpret_cast<const float3*>(mesh.normals.data);
    float3 n0 = normal_buffer[i0];
    float3 n1 = normal_buffer[i1];
    float3 n2 = normal_buffer[i2];
    float3 object_normal = (1.0f - barycentrics.x - barycentrics.y) * n0 + barycentrics.x * n1
        + barycentrics.y * n2;

    float3 world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(object_normal));
    float3 ffnormal = faceforward(world_normal, -optixGetWorldRayDirection(), world_normal);
#if TOON_SHADING
    phongToonShade(phong.Kd, phong.Ka, phong.Ks, phong.Kr, phong.phong_exp, ffnormal);
#else
    phongShade(phong.Kd, phong.Ka, phong.Ks, phong.Kr, phong.phong_exp, ffnormal);
#endif
}

extern "C" __global__ void __closesthit__checker_radiance()
{
    const whitted::HitGroupData*      sbt_data = (whitted::HitGroupData*)optixGetSbtDataPointer();
    const MaterialData::CheckerPhong& checker  = sbt_data->material_data.checker;

    float3 Kd, Ka, Ks, Kr;
    float  phong_exp;

    float2 texcoord = make_float2( __uint_as_float( optixGetAttribute_3() ), __uint_as_float( optixGetAttribute_4() ) );
    float2 t        = texcoord * checker.inv_checker_size;
    t.x             = floorf( t.x );
    t.y             = floorf( t.y );

    int which_check = ( static_cast<int>( t.x ) + static_cast<int>( t.y ) ) & 1;

    if( which_check )
    {
        Kd        = checker.Kd1;
        Ka        = checker.Ka1;
        Ks        = checker.Ks1;
        Kr        = checker.Kr1;
        phong_exp = checker.phong_exp1;
    }
    else
    {
        Kd        = checker.Kd2;
        Ka        = checker.Ka2;
        Ks        = checker.Ks2;
        Kr        = checker.Kr2;
        phong_exp = checker.phong_exp2;
    }

    float3 object_normal = make_float3( __uint_as_float( optixGetAttribute_0() ), __uint_as_float( optixGetAttribute_1() ),
                                        __uint_as_float( optixGetAttribute_2() ) );
    float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace( object_normal ) );
    float3 ffnormal     = faceforward( world_normal, -optixGetWorldRayDirection(), world_normal );
    phongShade( Kd, Ka, Ks, Kr, phong_exp, ffnormal );
}

extern "C" __global__ void __closesthit__metal_radiance()
{
    const whitted::HitGroupData* sbt_data = (whitted::HitGroupData*)optixGetSbtDataPointer();
    const MaterialData::Phong&   phong    = sbt_data->material_data.metal;

    float3 object_normal = make_float3( __uint_as_float( optixGetAttribute_0() ), __uint_as_float( optixGetAttribute_1() ),
                                        __uint_as_float( optixGetAttribute_2() ) );

    float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace( object_normal ) );
    float3 ffnormal     = faceforward( world_normal, -optixGetWorldRayDirection(), world_normal );
    phongShade( phong.Kd, phong.Ka, phong.Ks, phong.Kr, phong.phong_exp, ffnormal );
}

extern "C" __global__ void __closesthit__point_cloud_radiance()
{
    const whitted::HitGroupData* sbt_data = (whitted::HitGroupData*)optixGetSbtDataPointer();
    const MaterialData::Phong& phong = sbt_data->material_data.metal;

    float3 object_normal = make_float3(__uint_as_float(optixGetAttribute_0()), __uint_as_float(optixGetAttribute_1()),
        __uint_as_float(optixGetAttribute_2()));

    float3 world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(object_normal));
    float3 ffnormal = faceforward(world_normal, -optixGetWorldRayDirection(), world_normal);
    phongShadePointCloud(phong.Kd, phong.Ka, phong.Ks, phong.Kr, phong.phong_exp, ffnormal);
}

extern "C" __global__ void __closesthit__full_occlusion()
{
    phongShadowed();
}

extern "C" __global__ void __closesthit__glass_radiance()
{
    const whitted::HitGroupData* sbt_data = (whitted::HitGroupData*)optixGetSbtDataPointer();
    const MaterialData::Glass&   glass    = sbt_data->material_data.glass;

    whitted::PayloadRadiance prd_radiance = getPayloadRadiance();

    float3 object_normal = make_float3( __uint_as_float( optixGetAttribute_0() ), __uint_as_float( optixGetAttribute_1() ),
                                        __uint_as_float( optixGetAttribute_2() ) );
    object_normal        = normalize( object_normal );

    // intersection vectors
    const float3 n        = normalize( optixTransformNormalFromObjectToWorldSpace( object_normal ) );  // normal
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();  // incident direction
    const float  ray_t    = optixGetRayTmax();
    float3       t;  // transmission direction
    float3       r;  // reflection direction

    float3                             hit_point       = ray_orig + ray_t * ray_dir;
    GeometryData::SphereShell::HitType hit_type        = (GeometryData::SphereShell::HitType)optixGetHitKind();
    float3                             front_hit_point = hit_point, back_hit_point = hit_point;

    if( hit_type & GeometryData::SphereShell::HIT_OUTSIDE_FROM_OUTSIDE || hit_type & GeometryData::SphereShell::HIT_INSIDE_FROM_INSIDE )
    {
        front_hit_point += params.scene_epsilon * object_normal;
        back_hit_point -= params.scene_epsilon * object_normal;
    }
    else
    {
        front_hit_point -= params.scene_epsilon * object_normal;
        back_hit_point += params.scene_epsilon * object_normal;
    }

    const float3 fhp = optixTransformPointFromObjectToWorldSpace( front_hit_point );
    const float3 bhp = optixTransformPointFromObjectToWorldSpace( back_hit_point );

    float  reflection = 1.0f;
    float3 result     = make_float3( 0.0f );

    const int depth = prd_radiance.depth;

    float3 beer_attenuation;
    if( dot( n, ray_dir ) > 0 )
    {
        // Beer's law attenuation
        beer_attenuation = exp( glass.extinction_constant * ray_t );
    }
    else
    {
        beer_attenuation = make_float3( 1 );
    }

    // refraction
    // compare depth to max_depth - 1 to leave room for a potential shadow ray trace
    if( depth < min( glass.refraction_maxdepth, params.max_depth - 1 ) )
    {
        if( refract( t, ray_dir, n, glass.refraction_index ) )
        {
            // check for external or internal reflection
            float cos_theta = dot( ray_dir, n );
            if( cos_theta < 0.0f )
                cos_theta = -cos_theta;
            else
                cos_theta = dot( t, n );

            reflection = fresnel_schlick( cos_theta, glass.fresnel_exponent, glass.fresnel_minimum, glass.fresnel_maximum );

            float importance =
                prd_radiance.importance * ( 1.0f - reflection ) * luminance( glass.refraction_color * beer_attenuation );
            float3 color = glass.cutoff_color;
            if( importance > glass.importance_cutoff )
            {
                color = traceRadianceRay( bhp, t, depth + 1, importance );
            }
            result += ( 1.0f - reflection ) * glass.refraction_color * color;
        }
        // else TIR
    }  // else reflection==1 so refraction has 0 weight

    // reflection
    // compare depth to max_depth - 1 to leave room for a potential shadow ray trace
    float3 color = glass.cutoff_color;
    if( depth < min( glass.reflection_maxdepth, params.max_depth - 1 ) )
    {
        r = reflect( ray_dir, n );

        float importance = prd_radiance.importance * reflection * luminance( glass.reflection_color * beer_attenuation );
        if( importance > glass.importance_cutoff )
        {
            color = traceRadianceRay( fhp, r, depth + 1, importance );
        }
    }
    result += reflection * glass.reflection_color * color;

    result = result * beer_attenuation;

    prd_radiance.result = result;
    setPayloadRadiance( prd_radiance );
}

extern "C" __global__ void __anyhit__glass_occlusion()
{
    const whitted::HitGroupData* sbt_data = (whitted::HitGroupData*)optixGetSbtDataPointer();
    const MaterialData::Glass&   glass    = sbt_data->material_data.glass;

    float3 object_normal = make_float3( __uint_as_float( optixGetAttribute_0() ), __uint_as_float( optixGetAttribute_1() ),
                                        __uint_as_float( optixGetAttribute_2() ) );

    whitted::PayloadOcclusion shadow_prd = getPayloadOcclusion();

    float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace( object_normal ) );
    float  nDi          = fabs( dot( world_normal, optixGetWorldRayDirection() ) );

    shadow_prd.result *= 1 - fresnel_schlick( nDi, 5, 1 - glass.shadow_attenuation, make_float3( 1 ) );
    setPayloadOcclusion( shadow_prd );

    // Test the attenuation of the light from the glass shell
    if( luminance( shadow_prd.result ) < glass.importance_cutoff )
        // The attenuation is so high, > 99% blocked, that we can consider testing to be done.
        optixTerminateRay();
    else
        // There is still some light coming through the glass shell that we should test other occluders.
        // We "ignore" the intersection with the glass shell, meaning that shadow testing will continue.
        // If the ray does not hit another occluder, the light's attenuation from this glass shell
        // (along with other glass shells) is then used.
        optixIgnoreIntersection();
}

extern "C" __global__ void __miss__constant_bg()
{
    whitted::PayloadRadiance prd = getPayloadRadiance();
    prd.result                   = params.miss_color;
    setPayloadRadiance( prd );
}
