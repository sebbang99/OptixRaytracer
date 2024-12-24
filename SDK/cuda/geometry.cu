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

#include "helpers.h"
#include "whitted.h"
#include <float.h>
#include <iostream>

extern "C" __global__ void __intersection__cube()
{
    const whitted::HitGroupData* sbt_data = reinterpret_cast<whitted::HitGroupData*>(optixGetSbtDataPointer());
    const GeometryData::AABBs& cube = sbt_data->geometry_data.getAabb();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();

    float t1, t2, t_near = -FLT_MAX, t_far = FLT_MAX;
    float3 normal = make_float3(0.0f, 0.0f, 0.0f);

    // x-axis
    if (ray_dir.x == 0.0f) {
        if (ray_orig.x < cube.min.x || ray_orig.x > cube.max.x) return;
    }
    else {
        t1 = (cube.min.x - ray_orig.x) / ray_dir.x;
        t2 = (cube.max.x - ray_orig.x) / ray_dir.x;

        if (t1 > t2) {
            float tmp = t1;
            t1 = t2;
            t2 = tmp;
        }
        if (t1 > t_near) {
            t_near = t1;
            normal = make_float3(-1.0f, 0.0f, 0.0f);
        }
        if (t2 < t_far) {
            t_far = t2;
        }
        if (t_near > t_far) return;
        if (t_far < 0.0f || t_near > t_far) return;
    }

    // y-axis
    if (ray_dir.y == 0.0f) {
        if (ray_orig.y < cube.min.y || ray_orig.y > cube.max.y) return;
    }
    else {
        t1 = (cube.min.y - ray_orig.y) / ray_dir.y;
        t2 = (cube.max.y - ray_orig.y) / ray_dir.y;

        if (t1 > t2) {
            float tmp = t1;
            t1 = t2;
            t2 = tmp;
        }
        if (t1 > t_near) {
            t_near = t1;
            normal = make_float3(0.0f, 1.0f, 0.0f);
        }
        if (t2 < t_far) {
            t_far = t2;
        }
        if (t_near > t_far) return;
        if (t_far < 0.0f || t_near > t_far) return;
    }

    // z-axis
    if (ray_dir.z == 0.0f) {
        if (ray_orig.z < cube.min.z || ray_orig.z > cube.max.z) return;
    }
    else {
        t1 = (cube.min.z - ray_orig.z) / ray_dir.z;
        t2 = (cube.max.z - ray_orig.z) / ray_dir.z;

        if (t1 > t2) {
            float tmp = t1;
            t1 = t2;
            t2 = tmp;
        }
        if (t1 > t_near) {
            t_near = t1;
            normal = make_float3(0.0f, 0.0f, -1.0f);
        }
        if (t2 < t_far) {
            t_far = t2;
        }
        if (t_near > t_far) return;
        if (t_far < 0.0f || t_near > t_far) return;
    }

    if (t_near == -FLT_MAX)
        return;
    else
        optixReportIntersection(t_near, 0, float3_as_args(normal));
}

extern "C" __global__ void __intersection__cylinder()
{
    const whitted::HitGroupData* sbt_data = reinterpret_cast<whitted::HitGroupData*>(optixGetSbtDataPointer());
    const GeometryData::Cylinder& cylinder = sbt_data->geometry_data.getCylinder();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();

    const float3 center = cylinder.center;
    const float radius = cylinder.radius;
    const float height = cylinder.height;

    const float3 moved_orig = ray_orig - center;

    float k2 = 1.0f - ray_dir.y * ray_dir.y;
    float k1 = dot(moved_orig, ray_dir) - moved_orig.y * ray_dir.y;
    float k0 = dot(moved_orig, moved_orig) - moved_orig.y * moved_orig.y - radius * radius;

    float h = k1 * k1 - k2 * k0;
    if (h < 0.0f) return;

    h = sqrtf(h);
    float t = (-k1 - h) / k2;

    float y;
    // Check for intersection with cylinder body
    if (t >= ray_tmin && t <= ray_tmax) {
        y = moved_orig.y + t * ray_dir.y;
        if (y > -height && y < height) {
            float3 normal = (moved_orig + t * ray_dir - make_float3(0.0f, y, 0.0f)) / radius;
            optixReportIntersection(t, 0, float3_as_args(normal));
        }
    }

    // Check for intersection with caps
    t = (((y < 0.0f) ? -height : height) - moved_orig.y) / ray_dir.y;
    if (t >= ray_tmin && t <= ray_tmax) {
        if (abs(k1 + k2 * t) < h) {
            float3 normal = make_float3(0.0f, (y < 0.0f ? -1.0f : 1.0f), 0.0f);
            optixReportIntersection(t, 0, float3_as_args(normal));
        }
    }
}


extern "C" {
__constant__ whitted::LaunchParams params;
}

extern "C" __global__ void __intersection__parallelogram()
{
    const whitted::HitGroupData*       sbt_data = reinterpret_cast<whitted::HitGroupData*>( optixGetSbtDataPointer() );
    const GeometryData::Parallelogram& floor    = sbt_data->geometry_data.getParallelogram();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();

    float3 n  = make_float3( floor.plane );
    float  dt = dot( ray_dir, n );
    float  t  = ( floor.plane.w - dot( n, ray_orig ) ) / dt;
    if( t > ray_tmin && t < ray_tmax )
    {
        float3 p  = ray_orig + ray_dir * t;
        float3 vi = p - floor.anchor;
        float  a1 = dot( floor.v1, vi );
        if( a1 >= 0 && a1 <= 1 )
        {
            float a2 = dot( floor.v2, vi );
            if( a2 >= 0 && a2 <= 1 )
            {
                optixReportIntersection( t, 0, float3_as_args( n ), __float_as_uint( a1 ), __float_as_uint( a2 ) );
            }
        }
    }
}


extern "C" __global__ void __intersection__sphere_shell()
{
    const whitted::HitGroupData*     sbt_data = reinterpret_cast<whitted::HitGroupData*>( optixGetSbtDataPointer() );
    const GeometryData::SphereShell& sphere_shell = sbt_data->geometry_data.getSphereShell();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    const float  ray_tmin = optixGetRayTmin(), ray_tmax = optixGetRayTmax();

    float3 O = ray_orig - sphere_shell.center;
    float  l = 1 / length( ray_dir );
    float3 D = ray_dir * l;

    float b = dot( O, D ), sqr_b = b * b;
    float O_dot_O = dot( O, O );
    float radius1 = sphere_shell.radius1, radius2 = sphere_shell.radius2;
    float sqr_radius1 = radius1 * radius1, sqr_radius2 = radius2 * radius2;

    // check if we are outside of outer sphere
    if( O_dot_O > sqr_radius2 + params.scene_epsilon )
    {
        if( O_dot_O - sqr_b < sqr_radius2 - params.scene_epsilon )
        {
            float c    = O_dot_O - sqr_radius2;
            float root = sqr_b - c;
            if( root > 0.0f )
            {
                float  t      = -b - sqrtf( root );
                float3 normal = ( O + t * D ) / radius2;
                optixReportIntersection( t * l, GeometryData::SphereShell::HIT_OUTSIDE_FROM_OUTSIDE, float3_as_args( normal ) );
            }
        }
    }
    // else we are inside of the outer sphere
    else
    {
        float c    = O_dot_O - sqr_radius1;
        float root = b * b - c;
        if( root > 0.0f )
        {
            float t = -b - sqrtf( root );
            // do we hit inner sphere from between spheres?
            if( t * l > ray_tmin && t * l < ray_tmax )
            {
                float3 normal = ( O + t * D ) / ( -radius1 );
                optixReportIntersection( t * l, GeometryData::SphereShell::HIT_INSIDE_FROM_OUTSIDE, float3_as_args( normal ) );
            }
            else
            {
                // do we hit inner sphere from within both spheres?
                t = -b + ( root > 0 ? sqrtf( root ) : 0.f );
                if( t * l > ray_tmin && t * l < ray_tmax )
                {
                    float3 normal = ( O + t * D ) / ( -radius1 );
                    optixReportIntersection( t * l, GeometryData::SphereShell::HIT_INSIDE_FROM_INSIDE, float3_as_args( normal ) );
                }
                else
                {
                    // do we hit outer sphere from between spheres?
                    c             = O_dot_O - sqr_radius2;
                    root          = b * b - c;
                    t             = -b + ( root > 0 ? sqrtf( root ) : 0.f );
                    float3 normal = ( O + t * D ) / radius2;
                    optixReportIntersection( t * l, GeometryData::SphereShell::HIT_OUTSIDE_FROM_INSIDE, float3_as_args( normal ) );
                }
            }
        }
        else
        {
            // do we hit outer sphere from between spheres?
            c             = O_dot_O - sqr_radius2;
            root          = b * b - c;
            float  t      = -b + ( root > 0 ? sqrtf( root ) : 0.f );
            float3 normal = ( O + t * D ) / radius2;
            optixReportIntersection( t * l, GeometryData::SphereShell::HIT_OUTSIDE_FROM_INSIDE, float3_as_args( normal ) );
        }
    }
}
