#include "models.h"

void load_obj_file(const std::string& filename, std::vector<Vertex>& vertices, std::vector<Index>& indices) {
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