#pragma once
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <vector>

struct Vertex {
    float px, py, pz;
    float nx, ny, nz;
};

class Mesh {
public:
    void init(QOpenGLFunctions_3_3_Core* gl,
               const std::vector<Vertex>& verts,
               const std::vector<unsigned int>& idx);
    void draw(QOpenGLFunctions_3_3_Core* gl);

private:
    QOpenGLVertexArrayObject m_vao;
    QOpenGLBuffer m_vbo{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer m_ebo{QOpenGLBuffer::IndexBuffer};
    int m_indexCount = 0;
};

// Procedural geometry generators. All shapes are built in a canonical
// unit size so instances are placed with a translate*rotate*scale model matrix.
namespace MeshGen {
    // Cube spanning [-0.5, 0.5] on every axis.
    void cube(std::vector<Vertex>& v, std::vector<unsigned int>& idx);

    // Cylinder, radius 1, extends from y=0 (base) to y=1 (top).
    void cylinder(std::vector<Vertex>& v, std::vector<unsigned int>& idx, int segments);

    // Cone, base radius 1 at y=0, apex at y=1. segments=4 gives a pyramid (house roof).
    void cone(std::vector<Vertex>& v, std::vector<unsigned int>& idx, int segments);

    // Flat quad in the XZ plane at y=0, spanning [-0.5, 0.5] on X and Z, normal +Y.
    void plane(std::vector<Vertex>& v, std::vector<unsigned int>& idx);

    // Flat ring (annulus) in the XZ plane at the given height, used for the track surface.
    void ring(std::vector<Vertex>& v, std::vector<unsigned int>& idx,
              float innerRadius, float outerRadius, float y, int segments);
}
