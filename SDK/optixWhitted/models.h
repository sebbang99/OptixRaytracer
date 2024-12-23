#pragma once

#include "../support/tinyobj/tiny_obj_loader.h"
#include "../cuda/GeometryData.h"
#include "../cuda/BufferView.h"
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

struct Vertex {
    float pos[3];  // Position
    float norm[3]; // Normal
    float tex[2];  // Texture coordinates
};

struct Index {
    unsigned int v1, v2, v3; // Triangle indices
};

void load_obj_file(const std::string& filename, std::vector<Vertex>& vertices, std::vector<Index>& indices);