#include "Mesh.h"
#include <cmath>

void Mesh::init(QOpenGLFunctions_3_3_Core* gl,
                 const std::vector<Vertex>& verts,
                 const std::vector<unsigned int>& idx) {
    m_indexCount = static_cast<int>(idx.size());

    m_vao.create();
    m_vao.bind();

    m_vbo.create();
    m_vbo.bind();
    m_vbo.allocate(verts.data(), static_cast<int>(verts.size() * sizeof(Vertex)));

    m_ebo.create();
    m_ebo.bind();
    m_ebo.allocate(idx.data(), static_cast<int>(idx.size() * sizeof(unsigned int)));

    gl->glEnableVertexAttribArray(0);
    gl->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                               reinterpret_cast<void*>(offsetof(Vertex, px)));
    gl->glEnableVertexAttribArray(1);
    gl->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                               reinterpret_cast<void*>(offsetof(Vertex, nx)));

    m_vao.release();
    m_vbo.release();
    m_ebo.release();
}

void Mesh::draw(QOpenGLFunctions_3_3_Core* gl) {
    m_vao.bind();
    gl->glDrawElements(GL_TRIANGLES, m_indexCount, GL_UNSIGNED_INT, nullptr);
    m_vao.release();
}

namespace MeshGen {

static void quad(std::vector<Vertex>& v, std::vector<unsigned int>& idx,
                  Vertex a, Vertex b, Vertex c, Vertex d) {
    unsigned int base = static_cast<unsigned int>(v.size());
    v.push_back(a); v.push_back(b); v.push_back(c); v.push_back(d);
    idx.push_back(base); idx.push_back(base + 1); idx.push_back(base + 2);
    idx.push_back(base); idx.push_back(base + 2); idx.push_back(base + 3);
}

void cube(std::vector<Vertex>& v, std::vector<unsigned int>& idx) {
    // +Z
    quad(v, idx, {-0.5f,-0.5f, 0.5f, 0,0,1}, { 0.5f,-0.5f, 0.5f, 0,0,1},
                 { 0.5f, 0.5f, 0.5f, 0,0,1}, {-0.5f, 0.5f, 0.5f, 0,0,1});
    // -Z
    quad(v, idx, { 0.5f,-0.5f,-0.5f, 0,0,-1}, {-0.5f,-0.5f,-0.5f, 0,0,-1},
                 {-0.5f, 0.5f,-0.5f, 0,0,-1}, { 0.5f, 0.5f,-0.5f, 0,0,-1});
    // +X
    quad(v, idx, { 0.5f,-0.5f, 0.5f, 1,0,0}, { 0.5f,-0.5f,-0.5f, 1,0,0},
                 { 0.5f, 0.5f,-0.5f, 1,0,0}, { 0.5f, 0.5f, 0.5f, 1,0,0});
    // -X
    quad(v, idx, {-0.5f,-0.5f,-0.5f, -1,0,0}, {-0.5f,-0.5f, 0.5f, -1,0,0},
                 {-0.5f, 0.5f, 0.5f, -1,0,0}, {-0.5f, 0.5f,-0.5f, -1,0,0});
    // +Y
    quad(v, idx, {-0.5f, 0.5f, 0.5f, 0,1,0}, { 0.5f, 0.5f, 0.5f, 0,1,0},
                 { 0.5f, 0.5f,-0.5f, 0,1,0}, {-0.5f, 0.5f,-0.5f, 0,1,0});
    // -Y
    quad(v, idx, {-0.5f,-0.5f,-0.5f, 0,-1,0}, { 0.5f,-0.5f,-0.5f, 0,-1,0},
                 { 0.5f,-0.5f, 0.5f, 0,-1,0}, {-0.5f,-0.5f, 0.5f, 0,-1,0});
}

void cylinder(std::vector<Vertex>& v, std::vector<unsigned int>& idx, int segments) {
    // Sides (smooth normals, one seam duplicate for the wrap).
    unsigned int sideBase = static_cast<unsigned int>(v.size());
    for (int i = 0; i <= segments; ++i) {
        float a = 2.0f * float(M_PI) * i / segments;
        float cx = std::cos(a), sz = std::sin(a);
        v.push_back({cx, 0.0f, sz, cx, 0.0f, sz});
        v.push_back({cx, 1.0f, sz, cx, 0.0f, sz});
    }
    for (int i = 0; i < segments; ++i) {
        unsigned int b0 = sideBase + i * 2, t0 = b0 + 1;
        unsigned int b1 = sideBase + (i + 1) * 2, t1 = b1 + 1;
        idx.push_back(b0); idx.push_back(b1); idx.push_back(t1);
        idx.push_back(b0); idx.push_back(t1); idx.push_back(t0);
    }
    // Top cap.
    unsigned int topCenter = static_cast<unsigned int>(v.size());
    v.push_back({0, 1, 0, 0, 1, 0});
    unsigned int topRimBase = static_cast<unsigned int>(v.size());
    for (int i = 0; i <= segments; ++i) {
        float a = 2.0f * float(M_PI) * i / segments;
        v.push_back({std::cos(a), 1.0f, std::sin(a), 0, 1, 0});
    }
    for (int i = 0; i < segments; ++i) {
        idx.push_back(topCenter); idx.push_back(topRimBase + i); idx.push_back(topRimBase + i + 1);
    }
    // Bottom cap.
    unsigned int botCenter = static_cast<unsigned int>(v.size());
    v.push_back({0, 0, 0, 0, -1, 0});
    unsigned int botRimBase = static_cast<unsigned int>(v.size());
    for (int i = 0; i <= segments; ++i) {
        float a = 2.0f * float(M_PI) * i / segments;
        v.push_back({std::cos(a), 0.0f, std::sin(a), 0, -1, 0});
    }
    for (int i = 0; i < segments; ++i) {
        idx.push_back(botCenter); idx.push_back(botRimBase + i + 1); idx.push_back(botRimBase + i);
    }
}

void cone(std::vector<Vertex>& v, std::vector<unsigned int>& idx, int segments) {
    for (int i = 0; i < segments; ++i) {
        float a0 = 2.0f * float(M_PI) * i / segments;
        float a1 = 2.0f * float(M_PI) * (i + 1) / segments;
        float x0 = std::cos(a0), z0 = std::sin(a0);
        float x1 = std::cos(a1), z1 = std::sin(a1);

        // Flat-shaded side face: normal from the two base edges to the apex.
        float ex0 = x0 - 0.0f, ey0 = -1.0f, ez0 = z0 - 0.0f;
        float ex1 = x1 - 0.0f, ey1 = -1.0f, ez1 = z1 - 0.0f;
        float nx = ey0 * ez1 - ez0 * ey1;
        float ny = ez0 * ex1 - ex0 * ez1;
        float nz = ex0 * ey1 - ey0 * ex1;
        float len = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (len > 1e-6f) { nx /= len; ny /= len; nz /= len; }

        unsigned int base = static_cast<unsigned int>(v.size());
        v.push_back({0, 1, 0, nx, ny, nz});
        v.push_back({x0, 0, z0, nx, ny, nz});
        v.push_back({x1, 0, z1, nx, ny, nz});
        idx.push_back(base); idx.push_back(base + 1); idx.push_back(base + 2);
    }
    // Bottom cap.
    unsigned int botCenter = static_cast<unsigned int>(v.size());
    v.push_back({0, 0, 0, 0, -1, 0});
    unsigned int botRimBase = static_cast<unsigned int>(v.size());
    for (int i = 0; i <= segments; ++i) {
        float a = 2.0f * float(M_PI) * i / segments;
        v.push_back({std::cos(a), 0.0f, std::sin(a), 0, -1, 0});
    }
    for (int i = 0; i < segments; ++i) {
        idx.push_back(botCenter); idx.push_back(botRimBase + i + 1); idx.push_back(botRimBase + i);
    }
}

void plane(std::vector<Vertex>& v, std::vector<unsigned int>& idx) {
    quad(v, idx,
         {-0.5f, 0.0f,  0.5f, 0, 1, 0},
         { 0.5f, 0.0f,  0.5f, 0, 1, 0},
         { 0.5f, 0.0f, -0.5f, 0, 1, 0},
         {-0.5f, 0.0f, -0.5f, 0, 1, 0});
}

void ring(std::vector<Vertex>& v, std::vector<unsigned int>& idx,
          float innerRadius, float outerRadius, float y, int segments) {
    unsigned int base = static_cast<unsigned int>(v.size());
    for (int i = 0; i <= segments; ++i) {
        float a = 2.0f * float(M_PI) * i / segments;
        float cx = std::cos(a), sz = std::sin(a);
        v.push_back({cx * innerRadius, y, sz * innerRadius, 0, 1, 0});
        v.push_back({cx * outerRadius, y, sz * outerRadius, 0, 1, 0});
    }
    for (int i = 0; i < segments; ++i) {
        unsigned int i0 = base + i * 2, o0 = i0 + 1;
        unsigned int i1 = base + (i + 1) * 2, o1 = i1 + 1;
        idx.push_back(i0); idx.push_back(o0); idx.push_back(o1);
        idx.push_back(i0); idx.push_back(o1); idx.push_back(i1);
    }
}

} // namespace MeshGen
