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

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <cuda/whitted.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Record.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>
#include <iomanip>
#include <cstring>

#include <iostream>
#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <cstdlib>

const uint32_t OBJ_COUNT = 5;
// idx 0 : blue sphere
// idx 1 : sphere shell
// idx 2 : floor
// idx 3 : cube
// idx 4 : cylinder

const uint32_t POINT_CLOUD_COUNT = 2000;
const uint32_t POLYGON_COUNT = 2;
const uint32_t GAS_COUNT = 1 + POLYGON_COUNT + 1;

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

bool              resize_dirty  = false;
bool              minimized     = false;

// Camera state
bool              camera_changed = true;
sutil::Camera     camera;
sutil::Trackball  trackball;
const float move_speed = 0.1f;

// Mouse state
int32_t           mouse_button = -1;

const int         max_trace = 12;

const size_t count_records = whitted::RAY_TYPE_COUNT * (OBJ_COUNT + POLYGON_COUNT + POINT_CLOUD_COUNT);

//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------
typedef sutil::Record<whitted::HitGroupData> HitGroupRecord;
HitGroupRecord hitgroup_records[count_records];

struct Instance
{
    float transform[12];
};

void allocateAndCopyBuffer(CUdeviceptr& device_ptr, const void* host_data, size_t byte_size)
{
    cudaMalloc(reinterpret_cast<void**>(&device_ptr), byte_size);
    cudaMemcpy(reinterpret_cast<void*>(device_ptr), host_data, byte_size, cudaMemcpyHostToDevice);
}

void load_obj_file_to_triangle_mesh(const std::string& filename, GeometryData::TriangleMesh& mesh)
{
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filename)) {
        if (!reader.Error().empty()) {
            std::cerr << "Error: " << reader.Error() << "\n";
        }
        exit(EXIT_FAILURE);
    }

    if (!reader.Warning().empty()) {
        std::cout << "Warning: " << reader.Warning() << "\n";
    }

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    // Prepare temporary host buffers
    std::vector<uint32_t> indices;
    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> texcoords;
    std::vector<float4> colors; // Placeholder, since .obj doesn't have color data by default

    // Iterate over shapes and faces
    for (const auto& shape : shapes) {
        size_t index_offset = 0;

        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            size_t num_face_vertices = shape.mesh.num_face_vertices[f];
            if (num_face_vertices != 3) {
                std::cerr << "Only triangular faces are supported.\n";
                exit(EXIT_FAILURE);
            }

            for (size_t v = 0; v < num_face_vertices; ++v) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                // Add index
                indices.push_back(static_cast<uint32_t>(positions.size()));

                // Add position
                positions.push_back({
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]
                    });

                // Add normal
                if (idx.normal_index >= 0) {
                    normals.push_back({
                        attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]
                        });
                }

                // Add texture coordinate
                if (idx.texcoord_index >= 0) {
                    texcoords.push_back({
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        attrib.texcoords[2 * idx.texcoord_index + 1]
                        });
                }
                else {
                    texcoords.push_back({ 0.0f, 0.0f }); // Default texcoord
                }

                // Placeholder for color (e.g., white color for all vertices)
                colors.push_back({ 1.0f, 1.0f, 1.0f, 1.0f });
            }

            index_offset += num_face_vertices;
        }
    }

    // Allocate and copy data to GPU buffers
    allocateAndCopyBuffer(mesh.indices.data, indices.data(), indices.size() * sizeof(uint32_t));
    mesh.indices.count = static_cast<unsigned int>(indices.size());
    mesh.indices.byte_stride = sizeof(uint32_t);
    mesh.indices.elmt_byte_size = sizeof(uint32_t);

    allocateAndCopyBuffer(mesh.positions.data, positions.data(), positions.size() * sizeof(float3));
    mesh.positions.count = static_cast<unsigned int>(positions.size());
    mesh.positions.byte_stride = sizeof(float3);
    mesh.positions.elmt_byte_size = sizeof(float3);

    allocateAndCopyBuffer(mesh.normals.data, normals.data(), normals.size() * sizeof(float3));
    mesh.normals.count = static_cast<unsigned int>(normals.size());
    mesh.normals.byte_stride = sizeof(float3);
    mesh.normals.elmt_byte_size = sizeof(float3);

    //allocateAndCopyBuffer(mesh.texcoords.data, texcoords.data(), texcoords.size() * sizeof(float2));
    //mesh.texcoords.count = static_cast<unsigned int>(texcoords.size());
    //mesh.texcoords.byte_stride = sizeof(float2);
    //mesh.texcoords.elmt_byte_size = sizeof(float2);

    //allocateAndCopyBuffer(mesh.colors.data, colors.data(), colors.size() * sizeof(float4));
    //mesh.colors.count = static_cast<unsigned int>(colors.size());
    //mesh.colors.byte_stride = sizeof(float4);
    //mesh.colors.elmt_byte_size = sizeof(float4);
}

// for MyTriangleMesh
void load_obj_file(const std::string& filename, std::vector<Vertex>& vertices, std::vector<Index>& indices) {
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filename)) {  // call LoadObj() inside here.
        if (!reader.Error().empty()) {
            std::cerr << "Error: " << reader.Error() << "\n";
        }
        exit(EXIT_FAILURE);
    }

    if (!reader.Warning().empty()) {
        std::cout << "Warning: " << reader.Warning() << "\n";
    }

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    // Iterate over shapes (each shape contains multiple faces)
    for (const auto& shape : shapes) {
        size_t index_offset = 0;

        // Iterate over faces
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            size_t num_face_vertices = shape.mesh.num_face_vertices[f];
            if (num_face_vertices != 3) {
                std::cerr << "Only triangular faces are supported.\n";
                exit(EXIT_FAILURE);
            }

            Index index;
            for (size_t v = 0; v < num_face_vertices; ++v) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                Vertex vertex = {};

                // Get vertex position
                vertex.pos[0] = attrib.vertices[3 * idx.vertex_index + 0];
                vertex.pos[1] = attrib.vertices[3 * idx.vertex_index + 1];
                vertex.pos[2] = attrib.vertices[3 * idx.vertex_index + 2];

                // Get vertex normal
                if (idx.normal_index >= 0) {
                    vertex.norm[0] = attrib.normals[3 * idx.normal_index + 0];
                    vertex.norm[1] = attrib.normals[3 * idx.normal_index + 1];
                    vertex.norm[2] = attrib.normals[3 * idx.normal_index + 2];
                }

                // Get texture coordinates
                if (idx.texcoord_index >= 0) {
                    vertex.tex[0] = attrib.texcoords[2 * idx.texcoord_index + 0];
                    vertex.tex[1] = attrib.texcoords[2 * idx.texcoord_index + 1];
                }

                // Add vertex to list
                vertices.push_back(vertex);

                // Add index
                if (v == 0) index.v1 = static_cast<unsigned int>(vertices.size() - 1);
                if (v == 1) index.v2 = static_cast<unsigned int>(vertices.size() - 1);
                if (v == 2) index.v3 = static_cast<unsigned int>(vertices.size() - 1);
            }
            indices.push_back(index);
            index_offset += num_face_vertices;
        }
    }
}

void load_obj_file_for_point_cloud(const std::string& filename, std::vector<Vertex>& vertices) {
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filename)) {  // call LoadObj() inside here.
        if (!reader.Error().empty()) {
            std::cerr << "Error: " << reader.Error() << "\n";
        }
        exit(EXIT_FAILURE);
    }

    if (!reader.Warning().empty()) {
        std::cout << "Warning: " << reader.Warning() << "\n";
    }

    const auto& attrib = reader.GetAttrib();

    for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
        Vertex vertex = {};
        vertex.pos[0] = attrib.vertices[i + 0];
        vertex.pos[1] = attrib.vertices[i + 1];
        vertex.pos[2] = attrib.vertices[i + 2];

        vertex.norm[0] = 0.0f;
        vertex.norm[1] = 0.0f;
        vertex.norm[2] = 0.0f;

        vertex.tex[0] = 0.0f;
        vertex.tex[1] = 0.0f;

        vertices.push_back(vertex);
    }
}

struct WhittedState
{
    OptixDeviceContext          context                   = 0;

    OptixTraversableHandle      gas_handle_aabb                = {};
    CUdeviceptr                 d_gas_output_buffer_aabb       = {};
    OptixTraversableHandle      gas_handle_triangle            = {};
    CUdeviceptr                 d_gas_output_buffer_triangle   = {};
    OptixTraversableHandle      gas_handle_triangle_wolf = {};
    CUdeviceptr                 d_gas_output_buffer_triangle_wolf = {};
    OptixTraversableHandle      gas_handle_aabb_point_cloud = {};
    CUdeviceptr                 d_gas_output_buffer_aabb_point_cloud = {};

    OptixTraversableHandle      ias_handle                     = {};
    CUdeviceptr                 d_ias_output_buffer            = {};

    OptixModule                 geometry_module           = 0;
    OptixModule                 camera_module             = 0;
    OptixModule                 shading_module            = 0;
    OptixModule                 sphere_module             = 0;

    OptixProgramGroup           raygen_prog_group         = 0;
    OptixProgramGroup           radiance_miss_prog_group  = 0;
    OptixProgramGroup           occlusion_miss_prog_group = 0;
    OptixProgramGroup           radiance_glass_sphere_prog_group  = 0;
    OptixProgramGroup           occlusion_glass_sphere_prog_group = 0;
    OptixProgramGroup           radiance_metal_sphere_prog_group  = 0;
    OptixProgramGroup           occlusion_metal_sphere_prog_group = 0;
    OptixProgramGroup           radiance_floor_prog_group         = 0;
    OptixProgramGroup           occlusion_floor_prog_group        = 0;
    // Sehee added begin
    OptixProgramGroup           radiance_cube_prog_group = 0;
    OptixProgramGroup           occlusion_cube_prog_group = 0;
    OptixProgramGroup           radiance_cylinder_prog_group = 0;
    OptixProgramGroup           occlusion_cylinder_prog_group = 0;
    OptixProgramGroup           radiance_cow_prog_group = 0;
    OptixProgramGroup           occlusion_cow_prog_group = 0;
    OptixProgramGroup           radiance_wolf_prog_group = 0;
    OptixProgramGroup           occlusion_wolf_prog_group = 0;
    OptixProgramGroup           radiance_point_cloud_prog_group = 0;
    OptixProgramGroup           occlusion_point_cloud_prog_group = 0;
    // Sehee added end

    OptixPipeline               pipeline                  = 0;
    OptixPipelineCompileOptions pipeline_compile_options  = {};

    CUstream                    stream                    = 0;
    whitted::LaunchParams       params;
    whitted::LaunchParams*      d_params                  = nullptr;

    OptixShaderBindingTable     sbt                       = {};
};

//------------------------------------------------------------------------------
//
//  Geometry and Camera data
//
//------------------------------------------------------------------------------

// Metal sphere, glass sphere, floor by default
const GeometryData::Sphere g_sphere = {
    { 2.0f, 1.5f, -2.5f }, // center
    1.0f                   // radius
};
const GeometryData::SphereShell g_sphere_shell = {
    { -7.0f, 2.3f, -4.0f }, // center
    0.96f,                 // radius1
    1.0f                   // radius2
};
const GeometryData::Parallelogram g_floor(
    make_float3( 32.0f, 0.0f, 0.0f ),    // v1
    make_float3( 0.0f, 0.0f, 16.0f ),    // v2
    make_float3( -16.0f, 0.01f, -8.0f )  // anchor
    );
const GeometryData::AABBs cube = {
    { -12.0f, 0.0f, -2.0f },    // min
    { -6.0f, 6.0f, 4.0f }       // max
};
const GeometryData::Cylinder cylinder = {
    make_float3(5.0f, 0.5f, 0.0f),  // center
    2.0f,                           // radius
    0.5f                            // height
};
GeometryData::Sphere *point_cloud;
//GeometryData::MyTriangleMesh cow;
//GeometryData::MyTriangleMesh wolf;
GeometryData::TriangleMesh cow;
GeometryData::TriangleMesh wolf;

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos( window, &xpos, &ypos );

    if( action == GLFW_PRESS )
    {
        mouse_button = button;
        trackball.startTracking(static_cast<int>( xpos ), static_cast<int>( ypos ));
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    whitted::LaunchParams* params = static_cast<whitted::LaunchParams*>( glfwGetWindowUserPointer( window ) );

    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        camera_changed = true;
    }
}


static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    // Keep rendering at the current resolution when the window is minimized.
    if( minimized )
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize( res_x, res_y );

    whitted::LaunchParams* params = static_cast<whitted::LaunchParams*>( glfwGetWindowUserPointer( window ) );
    params->width  = res_x;
    params->height = res_y;
    camera_changed = true;
    resize_dirty   = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_ESCAPE)
        {
            glfwSetWindowShouldClose(window, true);
        }
    }
    else if (key == GLFW_KEY_P) {
        float3 eye = camera.eye();
        std::cout << "*** Current camera position : (" << eye.x << ", " << eye.y << ", " << eye.z << ")\n";
        float3 lookat = camera.lookat();
        std::cout << "*** Current center position : (" << lookat.x << ", " << lookat.y << ", " << lookat.z << ")\n";
    }
    else if (key == GLFW_KEY_G)
    {
        std::cout << "G key pressed!\n";
    }
    else if (key == GLFW_KEY_W)
    {
        float3 eye = camera.eye();
        float3 lookat = camera.lookat();
        float3 W = normalize(lookat - eye);
        camera.setEye(eye + W * move_speed);
        camera.setLookat(lookat + W * move_speed);

        camera_changed = true;
    }
    else if (key == GLFW_KEY_A)
    {
        float3 eye = camera.eye();
        float3 lookat = camera.lookat();
        float3 W = normalize(lookat - eye);
        float3 U = normalize(cross(W, make_float3(0.0f, 1.0f, 0.0f)));
        camera.setEye(eye - U * move_speed);
        camera.setLookat(lookat - U * move_speed);

        camera_changed = true;
    }
    else if (key == GLFW_KEY_S)
    {
        float3 eye = camera.eye();
        float3 lookat = camera.lookat();
        float3 W = normalize(lookat - eye);
        camera.setEye(eye - W * move_speed);
        camera.setLookat(lookat - W * move_speed);

        camera_changed = true;
    }
    else if (key == GLFW_KEY_D)
    {
        float3 eye = camera.eye();
        float3 lookat = camera.lookat();
        float3 W = normalize(lookat - eye);
        float3 U = normalize(cross(W, make_float3(0.0f, 1.0f, 0.0f)));
        camera.setEye(eye + U * move_speed);
        camera.setLookat(lookat + U * move_speed);

        camera_changed = true;
    }
    else if (key == GLFW_KEY_E)
    {
        std::cout << "Up key pressed!\n";

        float3 eye = camera.eye();
        camera.setEye(make_float3(eye.x, eye.y + move_speed, eye.z));

        float3 lookat = camera.lookat();
        camera.setLookat(make_float3(lookat.x, lookat.y + move_speed, lookat.z));

        camera_changed = true;
    }
    else if (key == GLFW_KEY_Q)
    {
        std::cout << "Down key pressed!\n";

        float3 eye = camera.eye();
        camera.setEye(make_float3(eye.x, eye.y - move_speed, eye.z));

        float3 lookat = camera.lookat();
        camera.setLookat(make_float3(lookat.x, lookat.y - move_speed, lookat.z));

        camera_changed = true;
    }
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if(trackball.wheelEvent((int)yscroll))
        camera_changed = true;
}


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}

void initLaunchParams( WhittedState& state )
{
    const size_t accum_buffer_size = state.params.width * state.params.height * sizeof(float4);
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &state.params.accum_buffer ),
        accum_buffer_size
    ) );
    CUDA_CHECK( cudaMemset( reinterpret_cast<void*>( state.params.accum_buffer ), 0, accum_buffer_size ));

    state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    state.params.subframe_index = 0u;

    // Set ambient light color and point light position
    std::vector<Light> lights( 4 );
    // ambient
    lights[0].type            = Light::Type::AMBIENT;
    lights[0].ambient.color   = make_float3( 0.1f, 0.1f, 0.1f );

    // point
    lights[1].type            = Light::Type::POINT;
    lights[1].point.color     = make_float3( 0.5f, 0.5f, 0.5f );
    lights[1].point.intensity = 1.0f;
    lights[1].point.position  = make_float3( 60.0f, 40.0f, 0.0f );
    lights[1].point.falloff   = Light::Falloff::QUADRATIC;

    // spot
    lights[2].type = Light::Type::SPOT;
    lights[2].spot.color = make_float3(1.0f, 0.0f, 0.0f);
    lights[2].spot.intensity = 1.0f;
    lights[2].spot.position = make_float3(-9.0f, 10.0f, 1.0f);
    lights[2].spot.falloff = Light::Falloff::QUADRATIC;
    lights[2].spot.direction = make_float3(0.0f, -1.0f, 0.0f);
    lights[2].spot.cutoff = 30.0f;

    lights[3].type = Light::Type::SPOT;
    lights[3].spot.color = make_float3(1.0f, 1.0f, 1.0f);
    lights[3].spot.intensity = 1.0f;
    lights[3].spot.position = make_float3(4.0f, 4.0f, 0.0f);
    lights[3].spot.falloff = Light::Falloff::QUADRATIC;
    lights[3].spot.direction = make_float3(-1.0f, -1.0f, 0.0f);
    lights[3].spot.cutoff = 1.0f;

    state.params.lights.count = static_cast<unsigned int>( lights.size() );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.params.lights.data ), lights.size() * sizeof( Light ) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( state.params.lights.data ), lights.data(),
                            lights.size() * sizeof( Light ), cudaMemcpyHostToDevice ) );
    //state.params.miss_color = { 0.34f, 0.55f, 0.85f };    // back ground
    state.params.miss_color = { 0.070588f, 0.098039f, 0.250980f };

    state.params.max_depth = max_trace;
    state.params.scene_epsilon = 1.e-4f;

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( whitted::LaunchParams ) ) );

    state.params.handle = state.ias_handle;
}

inline OptixAabb sphere_bound( float3 center, float radius )
{
    float3 m_min = center - radius;
    float3 m_max = center + radius;

    return {
        m_min.x, m_min.y, m_min.z,
        m_max.x, m_max.y, m_max.z
    };
}


inline OptixAabb parallelogram_bound( float3 v1, float3 v2, float3 anchor )
{
    // v1 and v2 are scaled by 1./length^2.  Rescale back to normal for the bounds computation.
    const float3 tv1  = v1 / dot( v1, v1 );
    const float3 tv2  = v2 / dot( v2, v2 );
    const float3 p00  = anchor;
    const float3 p01  = anchor + tv1;
    const float3 p10  = anchor + tv2;
    const float3 p11  = anchor + tv1 + tv2;

    float3 m_min = fminf( fminf( p00, p01 ), fminf( p10, p11 ));
    float3 m_max = fmaxf( fmaxf( p00, p01 ), fmaxf( p10, p11 ));
    return {
        m_min.x, m_min.y, m_min.z,
        m_max.x, m_max.y, m_max.z
    };
}

inline OptixAabb cube_bound(float3 min, float3 max)
{
    float3 m_min = min;
    float3 m_max = max;

    return {
        m_min.x, m_min.y, m_min.z,
        m_max.x, m_max.y, m_max.z
    };
}

inline OptixAabb cylinder_bound(float3 center, float r, float h)
{
    float3 m_min = make_float3(center.x - r, center.y - h, center.z - r);
    float3 m_max = make_float3(center.x + r, center.y + h, center.z + r);

    return {
    m_min.x, m_min.y, m_min.z,
    m_max.x, m_max.y, m_max.z
    };
}

static void buildGas(
    const WhittedState &state,
    const OptixAccelBuildOptions &accel_options,
    const OptixBuildInput &build_input,
    OptixTraversableHandle &gas_handle,
    CUdeviceptr &d_gas_output_buffer
    )
{
    // Note : tempSizeInBytes has size of the memory temporarily used for building.
    OptixAccelBufferSizes gas_buffer_sizes;
    CUdeviceptr d_temp_buffer_gas;

    OPTIX_CHECK( optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &build_input,
        1,
        &gas_buffer_sizes));

    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &d_temp_buffer_gas ),
        gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output and size of compacted resulting GAS
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                compactedSizeOffset + 8
                ) );

    // Note : for emitted post-build properties
    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK( optixAccelBuild(
        state.context,
        0,  // stream
        &accel_options,
        &build_input,
        1,
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        &emitProperty,
        1) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( state.context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

void buildIas(WhittedState &state) {
    Instance instance = { {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0} };

    const size_t instance_size_in_bytes = sizeof(OptixInstance) * GAS_COUNT; 
    OptixInstance optix_instances[GAS_COUNT];
    memset(optix_instances, 0, instance_size_in_bytes);

    // AABB (sphere, shell, floor, cube, cylinder)
    optix_instances[0].flags = OPTIX_INSTANCE_FLAG_NONE;
    optix_instances[0].instanceId = 0;
    optix_instances[0].sbtOffset = 0; 
    optix_instances[0].visibilityMask = 1;
    optix_instances[0].traversableHandle = state.gas_handle_aabb;
    memcpy(optix_instances[0].transform, instance.transform, sizeof(float) * 12); 

    // My Triangle Mesh (cow)
    optix_instances[1].flags = OPTIX_INSTANCE_FLAG_NONE;
    optix_instances[1].instanceId = 1;
    optix_instances[1].sbtOffset = 10; 
    optix_instances[1].visibilityMask = 1; 
    optix_instances[1].traversableHandle = state.gas_handle_triangle;
    memcpy(optix_instances[1].transform, instance.transform, sizeof(float) * 12); 

    // My Triangle Mesh (wolf)
    optix_instances[2].flags = OPTIX_INSTANCE_FLAG_NONE;
    optix_instances[2].instanceId = 1;
    optix_instances[2].sbtOffset = 12;
    optix_instances[2].visibilityMask = 1;
    optix_instances[2].traversableHandle = state.gas_handle_triangle_wolf;
    memcpy(optix_instances[2].transform, instance.transform, sizeof(float) * 12);

    // AABB (point cloud)
    optix_instances[3].flags = OPTIX_INSTANCE_FLAG_NONE;
    optix_instances[3].instanceId = 0;
    optix_instances[3].sbtOffset = 14;
    optix_instances[3].visibilityMask = 1;
    optix_instances[3].traversableHandle = state.gas_handle_aabb_point_cloud;
    memcpy(optix_instances[3].transform, instance.transform, sizeof(float) * 12);

    CUdeviceptr d_instances;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instance_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_instances),
        optix_instances,
        instance_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    OptixBuildInput instance_input = {};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances = d_instances;
    instance_input.instanceArray.numInstances = GAS_COUNT;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &instance_input,
        1, 
        &ias_buffer_sizes
    ));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer),
        ias_buffer_sizes.tempSizeInBytes
    ));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.d_ias_output_buffer),
        ias_buffer_sizes.outputSizeInBytes
    ));

    OPTIX_CHECK(optixAccelBuild(
        state.context,
        0, 
        &accel_options,
        &instance_input,
        1, 
        d_temp_buffer,
        ias_buffer_sizes.tempSizeInBytes,
        state.d_ias_output_buffer,
        ias_buffer_sizes.outputSizeInBytes,
        &state.ias_handle,
        nullptr,
        0     
    ));


    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instances)));
}

void createGeometry( WhittedState &state )
{
    //
    // Build Custom Primitives
    //

    // Load AABB into device memory
    OptixAabb   aabb[OBJ_COUNT] = { sphere_bound( g_sphere.center, g_sphere.radius ),
                                  sphere_bound( g_sphere_shell.center, g_sphere_shell.radius2 ),
                                  parallelogram_bound( g_floor.v1, g_floor.v2, g_floor.anchor ),
                                            cube_bound(cube.min, cube.max),
                                            cylinder_bound(cylinder.center, cylinder.radius, cylinder.height)};
    CUdeviceptr d_aabb;

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb
        ), OBJ_COUNT * sizeof( OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_aabb ),
                &aabb,
                OBJ_COUNT * sizeof( OptixAabb ),
                cudaMemcpyHostToDevice
                ) );

    // Setup AABB build input
    uint32_t aabb_input_flags[] = {
        /* flags for metal sphere */
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        /* flag for glass sphere */
        OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL,
        /* flag for floor */
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        /* flag for cube */
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        /* flag for cylinder */
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
    };
    /* TODO: This API cannot control flags for different ray type */

    const uint32_t sbt_index[] = { 0, 1, 2, 3, 4 };
    CUdeviceptr    d_sbt_index;

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sbt_index ), sizeof(sbt_index) ) );
    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast<void*>( d_sbt_index ),
        sbt_index,
        sizeof( sbt_index ),
        cudaMemcpyHostToDevice ) );

    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb;
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags;
    aabb_input.customPrimitiveArray.numSbtRecords = OBJ_COUNT;
    aabb_input.customPrimitiveArray.numPrimitives = OBJ_COUNT;
    aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer         = d_sbt_index;
    aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes    = sizeof( uint32_t );
    aabb_input.customPrimitiveArray.primitiveIndexOffset         = 0;

    OptixAccelBuildOptions accel_options = {
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION,  // buildFlags
        OPTIX_BUILD_OPERATION_BUILD         // operation
    };

    buildGas(
        state,
        accel_options,
        aabb_input,
        state.gas_handle_aabb,
        state.d_gas_output_buffer_aabb);

    // These codes commented below are the version of using MyTriangleMesh geometry type.
    // Load triangle polygon model into device memory [cow]
    //std::vector<Vertex> vertices;
    //std::vector<Index> indices;

    //load_obj_file("../../../SDK/data/Cow/wuson.obj", vertices, indices);
    //cow.vertices = (Vertex *)malloc(vertices.size() * sizeof(Vertex));
    //memcpy(cow.vertices, vertices.data(), vertices.size() * sizeof(Vertex));
    //cow.indices = (Index*)malloc(indices.size() * sizeof(Index));
    //memcpy(cow.indices, indices.data(), indices.size() * sizeof(Index));
    //cow.vertices = &vertices[0];
    //cow.indices = &indices[0];

    //CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tri_vertex
    //    ), vertices.size() * sizeof(Vertex)));
    //CUDA_CHECK(cudaMemcpy(
    //    reinterpret_cast<void*>(d_tri_vertex),
    //    cow.vertices,
    //    vertices.size() * sizeof(Vertex),
    //    cudaMemcpyHostToDevice
    //));

    //CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tri_index
    //    ), indices.size() * sizeof(Index)));
    //CUDA_CHECK(cudaMemcpy(
    //    reinterpret_cast<void*>(d_tri_index),
    //    cow.indices,
    //    indices.size() * sizeof(Index),
    //    cudaMemcpyHostToDevice
    //));
    // vector of vertices and indices should be cleared.

    load_obj_file_to_triangle_mesh("../../../SDK/data/Cow/wuson.obj", cow);

    CUdeviceptr d_tri_vertex = cow.positions.data;
    CUdeviceptr d_tri_index = cow.indices.data;

    uint32_t common_input_flag[] = {
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
    };

    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexBuffers = &d_tri_vertex;
    triangle_input.triangleArray.numVertices = cow.positions.count;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangle_input.triangleArray.indexBuffer = d_tri_index;
    triangle_input.triangleArray.numIndexTriplets = cow.indices.count / 3; // same as the ret value of optixGetPrimitiveIndex().
    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;

    // Translation by -4 along z-axis
    float transform_matrix[12] = {
        1.0f, 0.0f, 0.0f, -9.0f,
        0.0f, 1.0f, 0.0f, 6.0f,
        0.0f, 0.0f, 1.0f, 1.0f
    };
    float* d_transform_matrix;
    cudaMalloc(reinterpret_cast<void**>(&d_transform_matrix), sizeof(transform_matrix));
    cudaMemcpy(d_transform_matrix, transform_matrix, sizeof(transform_matrix), cudaMemcpyHostToDevice);

    // Assign the transform matrix
    triangle_input.triangleArray.preTransform = reinterpret_cast<CUdeviceptr>(d_transform_matrix);

    triangle_input.triangleArray.flags = common_input_flag;
    triangle_input.triangleArray.numSbtRecords = 1;
    triangle_input.triangleArray.sbtIndexOffsetBuffer = NULL;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    //triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = a;
    triangle_input.triangleArray.primitiveIndexOffset = 0;
    triangle_input.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;

    buildGas(
        state,
        accel_options,
        triangle_input,
        state.gas_handle_triangle,
        state.d_gas_output_buffer_triangle);

    // Load triangle polygon model into device memory [wolf]
    load_obj_file_to_triangle_mesh("../../../SDK/data/Wolf/LowPolyWolf.obj", wolf);

    CUdeviceptr d_tri_vertex_wolf = wolf.positions.data;
    CUdeviceptr d_tri_index_wolf = wolf.indices.data;

    OptixBuildInput triangle_input_wolf = {};
    triangle_input_wolf.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input_wolf.triangleArray.vertexBuffers = &d_tri_vertex_wolf;
    triangle_input_wolf.triangleArray.numVertices = wolf.positions.count;
    triangle_input_wolf.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input_wolf.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangle_input_wolf.triangleArray.indexBuffer = d_tri_index_wolf;
    triangle_input_wolf.triangleArray.numIndexTriplets = wolf.indices.count / 3; // same as the ret value of optixGetPrimitiveIndex().
    triangle_input_wolf.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input_wolf.triangleArray.indexStrideInBytes = 12;
    triangle_input_wolf.triangleArray.flags = common_input_flag;
    triangle_input_wolf.triangleArray.numSbtRecords = 1;
    triangle_input_wolf.triangleArray.sbtIndexOffsetBuffer = NULL;
    triangle_input_wolf.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t) * 3;
    //triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = a;
    triangle_input_wolf.triangleArray.primitiveIndexOffset = 0;
    triangle_input_wolf.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;

    float transform_matrix_wolf[12] = {
        2.5f, 0.0f, 0.0f, 5.0f,
        0.0f, 2.5f, 0.0f, 1.0f,
        0.0f, 0.0f, 2.5f, 0.0f
    };
    float* d_transform_matrix_wolf;
    cudaMalloc(reinterpret_cast<void**>(&d_transform_matrix_wolf), sizeof(transform_matrix_wolf));
    cudaMemcpy(d_transform_matrix_wolf, transform_matrix_wolf, sizeof(transform_matrix_wolf), cudaMemcpyHostToDevice);

    // Assign the transform matrix
    triangle_input_wolf.triangleArray.preTransform = reinterpret_cast<CUdeviceptr>(d_transform_matrix_wolf);

    buildGas(
        state,
        accel_options,
        triangle_input_wolf,
        state.gas_handle_triangle_wolf,
        state.d_gas_output_buffer_triangle_wolf);

    // Load AABB into device memory
    std::vector<Vertex> pt_vertices;

    load_obj_file_for_point_cloud("../../../SDK/data/Cloud/Cloud_2000_uniform.obj", pt_vertices);

    const uint32_t point_count = pt_vertices.size();

    point_cloud = (GeometryData::Sphere*)malloc(point_count * sizeof(GeometryData::Sphere));
    OptixAabb* aabb_pc = (OptixAabb*)malloc(sizeof(OptixAabb) * point_count);
    CUdeviceptr d_aabb_pc;

    for (uint32_t i = 0; i < point_count; i++) {
        float3 center = make_float3(pt_vertices[i].pos[0] * 15.0f, pt_vertices[i].pos[1] * 5.0f + 5.0f, pt_vertices[i].pos[2] * 8.0f);

        point_cloud[i].center = center;
        point_cloud[i].radius = 0.015f;

        aabb_pc[i] = sphere_bound(center, 0.015f);
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_pc
        ), point_count * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_aabb_pc),
        aabb_pc,
        point_count * sizeof(OptixAabb),
        cudaMemcpyHostToDevice
    ));

    // Setup AABB build input
    uint32_t* aabb_input_flags_pc = (uint32_t*)malloc(sizeof(uint32_t) * point_count);
    for (uint32_t i = 0; i < point_count; i++)
        aabb_input_flags_pc[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    ///* TODO: This API cannot control flags for different ray type */

    uint32_t* sbt_index_pc = (uint32_t*)malloc(sizeof(uint32_t) * point_count);
    for (uint32_t i = 0; i < point_count; i++)
        sbt_index_pc[i] = i;

    CUdeviceptr    d_sbt_index_pc;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_index_pc), sizeof(uint32_t) * point_count));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_sbt_index_pc),
        sbt_index_pc,
        sizeof(uint32_t) * point_count,
        cudaMemcpyHostToDevice));

    OptixBuildInput aabb_input_pc = {};
    aabb_input_pc.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input_pc.customPrimitiveArray.aabbBuffers = &d_aabb_pc;
    aabb_input_pc.customPrimitiveArray.flags = aabb_input_flags_pc;
    aabb_input_pc.customPrimitiveArray.numSbtRecords = point_count;
    aabb_input_pc.customPrimitiveArray.numPrimitives = point_count;
    aabb_input_pc.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_index_pc;
    aabb_input_pc.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    aabb_input_pc.customPrimitiveArray.primitiveIndexOffset = 0;

    buildGas(
        state,
        accel_options,
        aabb_input_pc,
        state.gas_handle_aabb_point_cloud,
        state.d_gas_output_buffer_aabb_point_cloud);

    buildIas(state);

    CUDA_CHECK( cudaFree( (void*)d_aabb) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>(d_sbt_index) ) );
    // TODO : device memory free for triangle mesh.
}

void createModules( WhittedState &state )
{
    OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( nullptr, nullptr, "geometry.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreate(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            input,
            inputSize,
            LOG, &LOG_SIZE,
            &state.geometry_module ) );
    }

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( nullptr, nullptr, "camera.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreate(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            input,
            inputSize,
            LOG, &LOG_SIZE,
            &state.camera_module ) );
    }

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( nullptr, nullptr, "shading.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreate(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            input,
            inputSize,
            LOG, &LOG_SIZE,
            &state.shading_module ) );
    }

    {
        size_t      inputSize = 0;
        const char* input     = sutil::getInputData( nullptr, nullptr, "sphere.cu", inputSize );
        OPTIX_CHECK_LOG( optixModuleCreate(
            state.context,
            &module_compile_options,
            &state.pipeline_compile_options,
            input,
            inputSize,
            LOG, &LOG_SIZE,
            &state.sphere_module ) );
    }
}

static void createCameraProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           cam_prog_group;
    OptixProgramGroupOptions    cam_prog_group_options = {};
    OptixProgramGroupDesc       cam_prog_group_desc = {};
    cam_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    cam_prog_group_desc.raygen.module = state.camera_module;
    cam_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole_camera";

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &cam_prog_group_desc,
        1,
        &cam_prog_group_options,
        LOG, &LOG_SIZE,
        &cam_prog_group ) );

    program_groups.push_back(cam_prog_group);
    state.raygen_prog_group = cam_prog_group;
}

static void createGlassSphereProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_sphere_prog_group;
    OptixProgramGroupOptions    radiance_sphere_prog_group_options = {};
    OptixProgramGroupDesc       radiance_sphere_prog_group_desc = {};
    radiance_sphere_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_sphere_prog_group_desc.hitgroup.moduleIS            = state.geometry_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere_shell";
    radiance_sphere_prog_group_desc.hitgroup.moduleCH            = state.shading_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__glass_radiance";
    radiance_sphere_prog_group_desc.hitgroup.moduleAH            = nullptr;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &radiance_sphere_prog_group_desc,
        1,
        &radiance_sphere_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_sphere_prog_group ) );

    program_groups.push_back(radiance_sphere_prog_group);
    state.radiance_glass_sphere_prog_group = radiance_sphere_prog_group;

    OptixProgramGroup           occlusion_sphere_prog_group;
    OptixProgramGroupOptions    occlusion_sphere_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_sphere_prog_group_desc = {};
    occlusion_sphere_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_sphere_prog_group_desc.hitgroup.moduleIS            = state.geometry_module;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere_shell";
    occlusion_sphere_prog_group_desc.hitgroup.moduleCH            = nullptr;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
    occlusion_sphere_prog_group_desc.hitgroup.moduleAH            = state.shading_module;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__glass_occlusion";

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &occlusion_sphere_prog_group_desc,
        1,
        &occlusion_sphere_prog_group_options,
        LOG, &LOG_SIZE,
        &occlusion_sphere_prog_group ) );

    program_groups.push_back(occlusion_sphere_prog_group);
    state.occlusion_glass_sphere_prog_group = occlusion_sphere_prog_group;
}

static void createMetalSphereProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_sphere_prog_group;
    OptixProgramGroupOptions    radiance_sphere_prog_group_options = {};
    OptixProgramGroupDesc       radiance_sphere_prog_group_desc = {};
    radiance_sphere_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        radiance_sphere_prog_group_desc.hitgroup.moduleIS           = state.sphere_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__sphere";
    radiance_sphere_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__metal_radiance";
    radiance_sphere_prog_group_desc.hitgroup.moduleAH               = nullptr;
    radiance_sphere_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &radiance_sphere_prog_group_desc,
        1,
        &radiance_sphere_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_sphere_prog_group ) );

    program_groups.push_back(radiance_sphere_prog_group);
    state.radiance_metal_sphere_prog_group = radiance_sphere_prog_group;

    OptixProgramGroup           occlusion_sphere_prog_group;
    OptixProgramGroupOptions    occlusion_sphere_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_sphere_prog_group_desc = {};
    occlusion_sphere_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        occlusion_sphere_prog_group_desc.hitgroup.moduleIS           = state.sphere_module;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__sphere";
    occlusion_sphere_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__full_occlusion";
    occlusion_sphere_prog_group_desc.hitgroup.moduleAH               = nullptr;
    occlusion_sphere_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &occlusion_sphere_prog_group_desc,
        1,
        &occlusion_sphere_prog_group_options,
        LOG, &LOG_SIZE,
        &occlusion_sphere_prog_group ) );

    program_groups.push_back(occlusion_sphere_prog_group);
    state.occlusion_metal_sphere_prog_group = occlusion_sphere_prog_group;
}

static void createFloorProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroup           radiance_floor_prog_group;
    OptixProgramGroupOptions    radiance_floor_prog_group_options = {};
    OptixProgramGroupDesc       radiance_floor_prog_group_desc = {};
    radiance_floor_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_floor_prog_group_desc.hitgroup.moduleIS               = state.geometry_module;
    radiance_floor_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__parallelogram";
    radiance_floor_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    radiance_floor_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__checker_radiance";
    radiance_floor_prog_group_desc.hitgroup.moduleAH               = nullptr;
    radiance_floor_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &radiance_floor_prog_group_desc,
        1,
        &radiance_floor_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_floor_prog_group ) );

    program_groups.push_back(radiance_floor_prog_group);
    state.radiance_floor_prog_group = radiance_floor_prog_group;

    OptixProgramGroup           occlusion_floor_prog_group;
    OptixProgramGroupOptions    occlusion_floor_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_floor_prog_group_desc = {};
    occlusion_floor_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_floor_prog_group_desc.hitgroup.moduleIS               = state.geometry_module;
    occlusion_floor_prog_group_desc.hitgroup.entryFunctionNameIS    = "__intersection__parallelogram";
    occlusion_floor_prog_group_desc.hitgroup.moduleCH               = state.shading_module;
    occlusion_floor_prog_group_desc.hitgroup.entryFunctionNameCH    = "__closesthit__full_occlusion";
    occlusion_floor_prog_group_desc.hitgroup.moduleAH               = nullptr;
    occlusion_floor_prog_group_desc.hitgroup.entryFunctionNameAH    = nullptr;

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &occlusion_floor_prog_group_desc,
        1,
        &occlusion_floor_prog_group_options,
        LOG, &LOG_SIZE,
        &occlusion_floor_prog_group ) );

    program_groups.push_back(occlusion_floor_prog_group);
    state.occlusion_floor_prog_group = occlusion_floor_prog_group;
}

static void createCubeProgram(WhittedState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup           radiance_cube_prog_group;
    OptixProgramGroupOptions    radiance_cube_prog_group_options = {};
    OptixProgramGroupDesc       radiance_cube_prog_group_desc = {};
    radiance_cube_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_cube_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
    radiance_cube_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cube";
    radiance_cube_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    radiance_cube_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__aabb_radiance";
    radiance_cube_prog_group_desc.hitgroup.moduleAH = nullptr;
    radiance_cube_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &radiance_cube_prog_group_desc,
        1,
        &radiance_cube_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_cube_prog_group));

    program_groups.push_back(radiance_cube_prog_group);
    state.radiance_cube_prog_group = radiance_cube_prog_group;

    OptixProgramGroup           occlusion_cube_prog_group;
    OptixProgramGroupOptions    occlusion_cube_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_cube_prog_group_desc = {};
    occlusion_cube_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_cube_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
    occlusion_cube_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cube";
    occlusion_cube_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    occlusion_cube_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__full_occlusion";
    occlusion_cube_prog_group_desc.hitgroup.moduleAH = nullptr;
    occlusion_cube_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &occlusion_cube_prog_group_desc,
        1,
        &occlusion_cube_prog_group_options,
        LOG, &LOG_SIZE,
        &occlusion_cube_prog_group));

    program_groups.push_back(occlusion_cube_prog_group);
    state.occlusion_cube_prog_group = occlusion_cube_prog_group;
}

static void createCylinderProgram(WhittedState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup           radiance_cylinder_prog_group;
    OptixProgramGroupOptions    radiance_cylinder_prog_group_options = {};
    OptixProgramGroupDesc       radiance_cylinder_prog_group_desc = {};
    radiance_cylinder_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_cylinder_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
    radiance_cylinder_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cylinder";
    radiance_cylinder_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    radiance_cylinder_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__aabb_radiance";
    radiance_cylinder_prog_group_desc.hitgroup.moduleAH = nullptr;
    radiance_cylinder_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &radiance_cylinder_prog_group_desc,
        1,
        &radiance_cylinder_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_cylinder_prog_group));

    program_groups.push_back(radiance_cylinder_prog_group);
    state.radiance_cylinder_prog_group = radiance_cylinder_prog_group;

    OptixProgramGroup           occlusion_cylinder_prog_group;
    OptixProgramGroupOptions    occlusion_cylinder_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_cylinder_prog_group_desc = {};
    occlusion_cylinder_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_cylinder_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
    occlusion_cylinder_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cylinder";
    occlusion_cylinder_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    occlusion_cylinder_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__full_occlusion";
    occlusion_cylinder_prog_group_desc.hitgroup.moduleAH = nullptr;
    occlusion_cylinder_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &occlusion_cylinder_prog_group_desc,
        1,
        &occlusion_cylinder_prog_group_options,
        LOG, &LOG_SIZE,
        &occlusion_cylinder_prog_group));

    program_groups.push_back(occlusion_cylinder_prog_group);
    state.occlusion_cylinder_prog_group = occlusion_cylinder_prog_group;
}

static void createCowProgram(WhittedState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup           radiance_cow_prog_group;
    OptixProgramGroupOptions    radiance_cow_prog_group_options = {};
    OptixProgramGroupDesc       radiance_cow_prog_group_desc = {};
    radiance_cow_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    // No .moduleIS because it has has built-in intersection shader.
    radiance_cow_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    radiance_cow_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__mesh_radiance";
    radiance_cow_prog_group_desc.hitgroup.moduleAH = nullptr;
    radiance_cow_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &radiance_cow_prog_group_desc,
        1,
        &radiance_cow_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_cow_prog_group));

    program_groups.push_back(radiance_cow_prog_group);
    state.radiance_cow_prog_group = radiance_cow_prog_group;

    OptixProgramGroup           occlusion_cow_prog_group;
    OptixProgramGroupOptions    occlusion_cow_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_cow_prog_group_desc = {};
    occlusion_cow_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_cow_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    occlusion_cow_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__full_occlusion";
    occlusion_cow_prog_group_desc.hitgroup.moduleAH = nullptr;
    occlusion_cow_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &occlusion_cow_prog_group_desc,
        1,
        &occlusion_cow_prog_group_options,
        LOG, &LOG_SIZE,
        &occlusion_cow_prog_group));

    program_groups.push_back(occlusion_cow_prog_group);
    state.occlusion_cow_prog_group = occlusion_cow_prog_group;
}

static void createWolfProgram(WhittedState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup           radiance_wolf_prog_group;
    OptixProgramGroupOptions    radiance_wolf_prog_group_options = {};
    OptixProgramGroupDesc       radiance_wolf_prog_group_desc = {};
    radiance_wolf_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    // No .moduleIS because it has has built-in intersection shader.
    radiance_wolf_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    radiance_wolf_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__mesh_radiance";
    radiance_wolf_prog_group_desc.hitgroup.moduleAH = nullptr;
    radiance_wolf_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &radiance_wolf_prog_group_desc,
        1,
        &radiance_wolf_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_wolf_prog_group));

    program_groups.push_back(radiance_wolf_prog_group);
    state.radiance_wolf_prog_group = radiance_wolf_prog_group;

    OptixProgramGroup           occlusion_wolf_prog_group;
    OptixProgramGroupOptions    occlusion_wolf_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_wolf_prog_group_desc = {};
    occlusion_wolf_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_wolf_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    occlusion_wolf_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__full_occlusion";
    occlusion_wolf_prog_group_desc.hitgroup.moduleAH = nullptr;
    occlusion_wolf_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &occlusion_wolf_prog_group_desc,
        1,
        &occlusion_wolf_prog_group_options,
        LOG, &LOG_SIZE,
        &occlusion_wolf_prog_group));

    program_groups.push_back(occlusion_wolf_prog_group);
    state.occlusion_wolf_prog_group = occlusion_wolf_prog_group;
}

static void createPointCloudProgram(WhittedState& state, std::vector<OptixProgramGroup>& program_groups)
{
    OptixProgramGroup           radiance_point_cloud_prog_group;
    OptixProgramGroupOptions    radiance_point_cloud_prog_group_options = {};
    OptixProgramGroupDesc       radiance_point_cloud_prog_group_desc = {};
    radiance_point_cloud_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    radiance_point_cloud_prog_group_desc.hitgroup.moduleIS = state.sphere_module;
    radiance_point_cloud_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    radiance_point_cloud_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    radiance_point_cloud_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__point_cloud_radiance";
    radiance_point_cloud_prog_group_desc.hitgroup.moduleAH = nullptr;
    radiance_point_cloud_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &radiance_point_cloud_prog_group_desc,
        1,
        &radiance_point_cloud_prog_group_options,
        LOG, &LOG_SIZE,
        &radiance_point_cloud_prog_group));

    program_groups.push_back(radiance_point_cloud_prog_group);
    state.radiance_point_cloud_prog_group = radiance_point_cloud_prog_group;

    OptixProgramGroup           occlusion_point_cloud_prog_group;
    OptixProgramGroupOptions    occlusion_point_cloud_prog_group_options = {};
    OptixProgramGroupDesc       occlusion_point_cloud_prog_group_desc = {};
    occlusion_point_cloud_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    occlusion_point_cloud_prog_group_desc.hitgroup.moduleIS = state.sphere_module;
    occlusion_point_cloud_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    // shadow is not need for snow effect. 
    //occlusion_point_cloud_prog_group_desc.hitgroup.moduleCH = state.shading_module;
    //occlusion_point_cloud_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__point_cloud_occlusion";
    occlusion_point_cloud_prog_group_desc.hitgroup.moduleAH = nullptr;
    occlusion_point_cloud_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &occlusion_point_cloud_prog_group_desc,
        1,
        &occlusion_point_cloud_prog_group_options,
        LOG, &LOG_SIZE,
        &occlusion_point_cloud_prog_group));

    program_groups.push_back(occlusion_point_cloud_prog_group);
    state.occlusion_point_cloud_prog_group = occlusion_point_cloud_prog_group;
}

static void createMissProgram( WhittedState &state, std::vector<OptixProgramGroup> &program_groups )
{
    OptixProgramGroupOptions    miss_prog_group_options = {};
    OptixProgramGroupDesc       miss_prog_group_desc = {};
    miss_prog_group_desc.kind   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module             = state.shading_module;
    miss_prog_group_desc.miss.entryFunctionName  = "__miss__constant_bg";

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1,
        &miss_prog_group_options,
        LOG, &LOG_SIZE,
        &state.radiance_miss_prog_group ) );

    program_groups.push_back(state.radiance_miss_prog_group);

    miss_prog_group_desc.miss = {
        nullptr,    // module
        nullptr     // entryFunctionName
    };
    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1,
        &miss_prog_group_options,
        LOG, &LOG_SIZE,
        &state.occlusion_miss_prog_group ) );

    program_groups.push_back(state.occlusion_miss_prog_group);
}

void createPipeline( WhittedState &state )
{
    std::vector<OptixProgramGroup> program_groups;

    state.pipeline_compile_options = {
        false,                                                  // usesMotionBlur
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,          // traversableGraphFlags
        5,    /* RadiancePRD uses 5 payloads */                 // numPayloadValues
        5,    /* Parallelogram intersection uses 5 attrs */     // numAttributeValues
        OPTIX_EXCEPTION_FLAG_NONE,                              // exceptionFlags
        "params"                                                // pipelineLaunchParamsVariableName
    };

    // Prepare program groups
    createModules( state );
    createCameraProgram( state, program_groups );
    createGlassSphereProgram( state, program_groups );
    createMetalSphereProgram( state, program_groups );
    createFloorProgram( state, program_groups );
    // Sehee added begin
    createCubeProgram(state, program_groups);
    createCylinderProgram(state, program_groups);
    createCowProgram(state, program_groups);
    createWolfProgram(state, program_groups);
    createPointCloudProgram(state, program_groups);
    // Sehee added end
    createMissProgram( state, program_groups );

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = max_trace;
    OPTIX_CHECK_LOG( optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        static_cast<unsigned int>( program_groups.size() ),
        LOG, &LOG_SIZE,
        &state.pipeline ) );

    OptixStackSizes stack_sizes = {};
    for( auto& prog_group : program_groups )
    {
        OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes, state.pipeline ) );
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace,
                                             0,  // maxCCDepth
                                             0,  // maxDCDepth
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( state.pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            1  // maxTraversableDepth
                                            ) );
}

void createSBT( WhittedState &state )
{
    // Raygen program record
    {
        CUdeviceptr d_raygen_record;
        size_t      sizeof_raygen_record = sizeof( sutil::EmptyRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_raygen_record ),
            sizeof_raygen_record ) );

        sutil::EmptyRecord rg_sbt;
        optixSbtRecordPackHeader( state.raygen_prog_group, &rg_sbt );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_raygen_record ),
            &rg_sbt,
            sizeof_raygen_record,
            cudaMemcpyHostToDevice
        ) );

        state.sbt.raygenRecord = d_raygen_record;
    }

    // Miss program record
    {
        CUdeviceptr d_miss_record;
        size_t sizeof_miss_record = sizeof( sutil::EmptyRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_miss_record ),
            sizeof_miss_record*whitted::RAY_TYPE_COUNT ) );

        sutil::EmptyRecord ms_sbt[whitted::RAY_TYPE_COUNT];
        optixSbtRecordPackHeader( state.radiance_miss_prog_group, &ms_sbt[0] );
        optixSbtRecordPackHeader( state.occlusion_miss_prog_group, &ms_sbt[1] );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_miss_record ),
            ms_sbt,
            sizeof_miss_record*whitted::RAY_TYPE_COUNT,
            cudaMemcpyHostToDevice
        ) );

        state.sbt.missRecordBase          = d_miss_record;
        state.sbt.missRecordCount         = whitted::RAY_TYPE_COUNT;
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>( sizeof_miss_record );
    }

    // Hitgroup program record
    {
        /*const size_t count_records = whitted::RAY_TYPE_COUNT * (OBJ_COUNT + POLYGON_COUNT + POINT_CLOUD_COUNT);*/
        /*HitGroupRecord hitgroup_records[count_records];*/

        // Note: Fill SBT record array the same order like AS is built.
        int sbt_idx = 0;

        // Metal Sphere
        OPTIX_CHECK( optixSbtRecordPackHeader(
            state.radiance_metal_sphere_prog_group,
            &hitgroup_records[sbt_idx] ) );
        hitgroup_records[ sbt_idx ].data.geometry_data.setSphere( g_sphere );
        hitgroup_records[ sbt_idx ].data.material_data.metal = {
            { 0.2f, 0.5f, 0.5f },   // Ka
            { 0.2f, 0.7f, 0.8f },   // Kd
            { 0.9f, 0.9f, 0.9f },   // Ks
            { 0.5f, 0.5f, 0.5f },   // Kr
            64,                     // phong_exp
        };
        sbt_idx ++;

        OPTIX_CHECK( optixSbtRecordPackHeader(
            state.occlusion_metal_sphere_prog_group,
            &hitgroup_records[sbt_idx] ) );
        hitgroup_records[ sbt_idx ].data.geometry_data.setSphere( g_sphere );
        sbt_idx ++;

        // Glass Sphere
        OPTIX_CHECK( optixSbtRecordPackHeader(
            state.radiance_glass_sphere_prog_group,
            &hitgroup_records[sbt_idx] ) );
        hitgroup_records[ sbt_idx ].data.geometry_data.setSphereShell( g_sphere_shell );
        hitgroup_records[ sbt_idx ].data.material_data.glass = {
            1e-2f,                                  // importance_cutoff
            { 0.034f, 0.055f, 0.085f },             // cutoff_color
            3.0f,                                   // fresnel_exponent
            0.1f,                                   // fresnel_minimum
            1.0f,                                   // fresnel_maximum
            1.4f,                                   // refraction_index
            { 1.0f, 1.0f, 1.0f },                   // refraction_color
            { 1.0f, 1.0f, 1.0f },                   // reflection_color
            { logf(.83f), logf(.83f), logf(.83f) }, // extinction_constant
            { 0.6f, 0.6f, 0.6f },                   // shadow_attenuation
            10,                                     // refraction_maxdepth
            5                                       // reflection_maxdepth
        };
        sbt_idx ++;

        OPTIX_CHECK( optixSbtRecordPackHeader(
            state.occlusion_glass_sphere_prog_group,
            &hitgroup_records[sbt_idx] ) );
        hitgroup_records[ sbt_idx ].data.geometry_data.setSphereShell( g_sphere_shell );
        hitgroup_records[ sbt_idx ].data.material_data.glass.shadow_attenuation = { 0.6f, 0.6f, 0.6f };
        sbt_idx ++;

        // Floor
        OPTIX_CHECK( optixSbtRecordPackHeader(
            state.radiance_floor_prog_group,
            &hitgroup_records[sbt_idx] ) );
        hitgroup_records[ sbt_idx ].data.geometry_data.setParallelogram( g_floor );
        hitgroup_records[ sbt_idx ].data.material_data.checker = {
            { 0.172549f, 0.294118f, 0.203922f },      // Kd1
            { 0.325490f, 0.419608f, 0.203922f },     // Kd2
            { 0.172549f, 0.294118f, 0.203922f },      // Ka1
            { 0.325490f, 0.419608f, 0.203922f },     // Ka2
            { 0.0f, 0.0f, 0.0f },       // Ks1
            { 0.0f, 0.0f, 0.0f },       // Ks2
            { 0.0f, 0.0f, 0.0f },       // Kr1
            { 0.0f, 0.0f, 0.0f },       // Kr2
            0.0f,                       // phong_exp1
            0.0f,                       // phong_exp2
            { 32.0f, 16.0f }            // inv_checker_size
        };
        sbt_idx++;

        OPTIX_CHECK( optixSbtRecordPackHeader(
            state.occlusion_floor_prog_group,
            &hitgroup_records[sbt_idx] ) );
        hitgroup_records[ sbt_idx ].data.geometry_data.setParallelogram( g_floor );
        sbt_idx++;

        // Cube
        OPTIX_CHECK(optixSbtRecordPackHeader(
            state.radiance_cube_prog_group,
            &hitgroup_records[sbt_idx]));
        hitgroup_records[sbt_idx].data.geometry_data.setAabb(cube);
        hitgroup_records[sbt_idx].data.material_data.pink_mirror = {
            { 0.515f, 0.0215f, 0.0215f },   // Ka
            { 0.61424f, 0.03568f, 0.03568f },   // Kd
            { 0.633f, 0.727811f, 0.633 },   // Ks
            { 0.5f, 0.5f, 0.5f },   // Kr
            76.8f,                     // phong_exp
        };
        sbt_idx++;

        OPTIX_CHECK(optixSbtRecordPackHeader(
            state.occlusion_cube_prog_group,
            &hitgroup_records[sbt_idx]));
        hitgroup_records[sbt_idx].data.geometry_data.setAabb(cube);
        sbt_idx++;

        // Cylinder
        OPTIX_CHECK(optixSbtRecordPackHeader(
            state.radiance_cylinder_prog_group,
            &hitgroup_records[sbt_idx]));
        hitgroup_records[sbt_idx].data.geometry_data.setCylinder(cylinder);
        hitgroup_records[sbt_idx].data.material_data.pink_mirror = {
            { 0.515f, 0.0215f, 0.0215f },   // Ka
            { 0.61424f, 0.03568f, 0.03568f },   // Kd
            { 0.633f, 0.727811f, 0.633 },   // Ks
            { 0.5f, 0.5f, 0.5f },   // Kr
            76.8f,                     // phong_exp
        };
        sbt_idx++;

        OPTIX_CHECK(optixSbtRecordPackHeader(
            state.occlusion_cylinder_prog_group,
            &hitgroup_records[sbt_idx]));
        hitgroup_records[sbt_idx].data.geometry_data.setCylinder(cylinder);
        sbt_idx++;

        // Cow
        OPTIX_CHECK(optixSbtRecordPackHeader(
            state.radiance_cow_prog_group,
            &hitgroup_records[sbt_idx]));
        hitgroup_records[sbt_idx].data.geometry_data.setTriangleMesh(cow);
        hitgroup_records[sbt_idx].data.material_data.metal = {
            { 0.2f, 0.5f, 0.5f },   // Ka
            { 0.2f, 0.7f, 0.8f },   // Kd
            { 0.9f, 0.9f, 0.9f },   // Ks
            { 0.0f, 0.0f, 0.0f },   // Kr
            64,                     // phong_exp
        };
        sbt_idx++;

        OPTIX_CHECK(optixSbtRecordPackHeader(
            state.occlusion_cow_prog_group,
            &hitgroup_records[sbt_idx]));
        hitgroup_records[sbt_idx].data.geometry_data.setTriangleMesh(cow);
        sbt_idx++;

        // Wolf
        OPTIX_CHECK(optixSbtRecordPackHeader(
            state.radiance_wolf_prog_group,
            &hitgroup_records[sbt_idx]));
        hitgroup_records[sbt_idx].data.geometry_data.setTriangleMesh(wolf);
        hitgroup_records[sbt_idx].data.material_data.red_velvet = {
            { 0.1745f, 0.01175f, 0.01175f },   // Ka
            { 0.61424f, 0.34136f, 0.44136f },   // Kd
            { 0.727811f, 0.626959f, 0.626959f },   // Ks
            { 0.0f, 0.0f, 0.0f },   // Kr
            76.8,                     // phong_exp
        };
        sbt_idx++;

        OPTIX_CHECK(optixSbtRecordPackHeader(
            state.occlusion_wolf_prog_group,
            &hitgroup_records[sbt_idx]));
        hitgroup_records[sbt_idx].data.geometry_data.setTriangleMesh(wolf);
        sbt_idx++;

        // Point cloud
        for (uint32_t i = 0; i < POINT_CLOUD_COUNT; i++) {
            // [0, 1]
            float rand_value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

            MaterialData::Phong snow_material;
            float r = fmod(rand_value * 1.0f, 1.0f);  // 0.0 ~ 1.0
            float g = fmod(rand_value * 0.8f + 0.2f, 1.0f); // 0.2 ~ 1.0
            float b = fmod(rand_value * 0.8f + 0.2f, 1.0f); // 0.2 ~ 1.0

            if (rand_value < 0.2f) {
                snow_material = {
                    { r, g, b },          
                    { r, g, b },           
                    { 1.0f, 1.0f, 1.0f }, 
                    { 1.0f, 1.0f, 1.0f },  
                    1000,                   
                };
            }
            else if (rand_value < 0.4f) {
                snow_material = {
                    { r, g, b },         
                    { r * 0.7f, g * 0.8f, b * 0.6f }, 
                    { 0.9f, 0.9f, 0.9f }, 
                    { 0.6f, 0.6f, 0.6f },   
                    128,                  
                };
            }
            else if (rand_value < 0.6f) {
                snow_material = {
                    { r * 0.9f, g * 0.4f, b * 0.1f }, 
                    { r * 0.8f, g * 0.5f, b * 0.2f }, 
                    { 1.0f, 1.0f, 1.0f }, 
                    { 0.5f, 0.5f, 0.5f },  
                    64,                    
                };
            }
            else if (rand_value < 0.8f) {
                snow_material = {
                    { r * 0.6f, g * 0.4f, b * 1.0f },  
                    { r * 0.5f, g * 0.5f, b * 0.9f },    
                    { 1.0f, 1.0f, 1.0f },  
                    { 0.5f, 0.7f, 1.0f },  
                    128,                  
                };
            }
            else if (rand_value < 0.9f) {
                snow_material = {
                    { 1.0f, 0.6f, 0.8f }, 
                    { 1.0f, 0.4f, 0.6f },   
                    { 0.9f, 0.9f, 0.9f },  
                    { 0.8f, 0.6f, 0.6f }, 
                    64,                  
                };
            }
            else {
                snow_material = {
                    { 1.0f, 1.0f, 0.0f },   
                    { 1.0f, 0.5f, 0.0f },  
                    { 0.6f, 0.6f, 0.6f },  
                    { 0.8f, 0.3f, 0.3f }, 
                    64,                  
                };
            }

            OPTIX_CHECK(optixSbtRecordPackHeader(
                state.radiance_point_cloud_prog_group,
                &hitgroup_records[sbt_idx]));
            hitgroup_records[sbt_idx].data.geometry_data.setSphere(point_cloud[i]);
            hitgroup_records[sbt_idx].data.material_data.metal = snow_material;
            sbt_idx++;

            OPTIX_CHECK(optixSbtRecordPackHeader(
                state.occlusion_point_cloud_prog_group,
                &hitgroup_records[sbt_idx]));
            hitgroup_records[sbt_idx].data.geometry_data.setSphere(point_cloud[i]);
            sbt_idx++;
        }

        CUdeviceptr d_hitgroup_records;
        size_t      sizeof_hitgroup_record = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc(
            reinterpret_cast<void**>( &d_hitgroup_records ),
            sizeof_hitgroup_record*count_records
        ) );

        CUDA_CHECK( cudaMemcpy(
            reinterpret_cast<void*>( d_hitgroup_records ),
            hitgroup_records,
            sizeof_hitgroup_record*count_records,
            cudaMemcpyHostToDevice
        ) );

        state.sbt.hitgroupRecordBase            = d_hitgroup_records;
        state.sbt.hitgroupRecordCount           = count_records;
        state.sbt.hitgroupRecordStrideInBytes   = static_cast<uint32_t>( sizeof_hitgroup_record );
    }
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

void createContext( WhittedState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    OptixDeviceContext context;
    CUcontext          cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    state.context = context;
}

//
//
//

void initCameraState()
{
    camera.setEye( make_float3(6.37364, 7.81653, 9.76937));
    camera.setLookat( make_float3(-3.5, 1.4, -2.4));
    camera.setUp( make_float3( 0.0f, 1.0f, 0.0f ) );
    camera.setFovY( 60.0f );
    camera_changed = true;

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    trackball.setGimbalLock(true);
}

void handleCameraUpdate( WhittedState &state )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast<float>( state.params.width ) / static_cast<float>( state.params.height ) );
    state.params.eye = camera.eye();
    camera.UVWFrame( state.params.U, state.params.V, state.params.W );
}

void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer, whitted::LaunchParams& params )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( params.width, params.height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( params.accum_buffer ) ) );
    CUDA_CHECK( cudaMalloc(
        reinterpret_cast<void**>( &params.accum_buffer ),
        params.width*params.height*sizeof(float4)
    ) );
}

void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, WhittedState &state )
{
    // Update params on device
    if( camera_changed || resize_dirty )
        state.params.subframe_index = 0;

    handleCameraUpdate( state );
    handleResize( output_buffer, state.params );
}

void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, WhittedState& state )
{

    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( state.d_params ),
                                 &state.params,
                                 sizeof( whitted::LaunchParams ),
                                 cudaMemcpyHostToDevice,
                                 state.stream
    ) );

    OPTIX_CHECK( optixLaunch(
        state.pipeline,
        state.stream,
        reinterpret_cast<CUdeviceptr>( state.d_params ),
        sizeof( whitted::LaunchParams ),
        &state.sbt,
        state.params.width,  // launch width
        state.params.height, // launch height
        1                    // launch depth
    ) );
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}


void displaySubframe(
    sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
    sutil::GLDisplay&                 gl_display,
    GLFWwindow*                       window )
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}


void cleanupState( WhittedState& state )
{
    OPTIX_CHECK( optixPipelineDestroy     ( state.pipeline                ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.raygen_prog_group       ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_metal_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.occlusion_metal_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_glass_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.occlusion_glass_sphere_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_miss_prog_group         ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.radiance_floor_prog_group        ) );
    OPTIX_CHECK( optixProgramGroupDestroy ( state.occlusion_floor_prog_group       ) );
    // Sehee added begin
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_cube_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_cube_prog_group));
    // Sehee added end
    OPTIX_CHECK( optixModuleDestroy       ( state.shading_module          ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.geometry_module         ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.camera_module           ) );
    OPTIX_CHECK( optixModuleDestroy       ( state.sphere_module           ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context                 ) );


    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.raygenRecord       ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.missRecordBase     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_gas_output_buffer_aabb    ) ) );
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer_triangle)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_ias_output_buffer)));
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.accum_buffer    ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.params.lights.data     ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( state.d_params               ) ) );
}

int main( int argc, char* argv[] )
{
    WhittedState state;
    state.params.width  = 768;
    state.params.height = 768;
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    //
    // Parse command line options
    //
    std::string outfile;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--no-gl-interop" )
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            outfile = argv[++i];
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            int w, h;
            sutil::parseDimensions( dims_arg.c_str(), w, h );
            state.params.width  = w;
            state.params.height = h;
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        initCameraState();

        //
        // Set up OptiX state
        //
        createContext  ( state );
        createGeometry  ( state );
        createPipeline ( state );
        createSBT      ( state );

        initLaunchParams( state );

        //
        // Render loop
        //
        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "120240282 SEHEE CHO", state.params.width, state.params.height );
            glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
            glfwSetCursorPosCallback    ( window, cursorPosCallback     );
            glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback          ( window, keyCallback           );
            glfwSetScrollCallback       ( window, scrollCallback        );
            glfwSetWindowUserPointer    ( window, &state.params         );

            {
                // output_buffer needs to be destroyed before cleanupUI is called
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                        output_buffer_type,
                        state.params.width,
                        state.params.height
                        );

                output_buffer.setStream( state.stream );
                sutil::GLDisplay gl_display;

                std::chrono::duration<double > state_update_time( 0.0 );
                std::chrono::duration<double> render_time( 0.0 );
                std::chrono::duration<double> display_time( 0.0 );

                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState( output_buffer, state );
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe( output_buffer, state );
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe( output_buffer, gl_display, window );
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats( state_update_time, render_time, display_time );

                    glfwSwapBuffers( window );

                    ++state.params.subframe_index;
                }
                while( !glfwWindowShouldClose( window ) );

            }
            sutil::cleanupUI( window );
        }
        else
        {
            if ( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                sutil::initGLFW(); // For GL context
                sutil::initGL();
            }

            {
                // this scope is for output_buffer, to ensure the destructor is called bfore glfwTerminate()

                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
                    );

                handleCameraUpdate( state );
                handleResize( output_buffer, state.params );
                launchSubframe( output_buffer, state );

                sutil::ImageBuffer buffer;
                buffer.data         = output_buffer.getHostPointer();
                buffer.width        = output_buffer.width();
                buffer.height       = output_buffer.height();
                buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
                sutil::saveImage( outfile.c_str(), buffer, false );
            }

            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                glfwTerminate();
            }
        }

        cleanupState( state );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
