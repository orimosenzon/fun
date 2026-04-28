#include "Surface.h"

#include <QtMath>
#include <cmath>

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;
}

Mesh Surface::tessellate() const {
    const int nU = uSamples();
    const int nV = vSamples();
    const float du = (uMax() - uMin()) / nU;
    const float dv = (vMax() - vMin()) / nV;
    const float h = 1e-3f;

    Mesh mesh;
    mesh.vertices.reserve((nU + 1) * (nV + 1));

    for (int i = 0; i <= nU; ++i) {
        const float u = uMin() + i * du;
        for (int j = 0; j <= nV; ++j) {
            const float v = vMin() + j * dv;
            QVector3D p = evaluate(u, v);
            QVector3D pu = (evaluate(u + h, v) - evaluate(u - h, v)) / (2 * h);
            QVector3D pv = (evaluate(u, v + h) - evaluate(u, v - h)) / (2 * h);
            QVector3D n = QVector3D::crossProduct(pu, pv);
            if (n.lengthSquared() > 1e-12f) n.normalize();
            mesh.vertices.push_back({p, n, u, v});
        }
    }

    mesh.indices.reserve(nU * nV * 6);
    auto idx = [nV](int i, int j) { return i * (nV + 1) + j; };
    for (int i = 0; i < nU; ++i) {
        for (int j = 0; j < nV; ++j) {
            const unsigned int a = idx(i,     j);
            const unsigned int b = idx(i + 1, j);
            const unsigned int c = idx(i + 1, j + 1);
            const unsigned int d = idx(i,     j + 1);
            mesh.indices.push_back(a); mesh.indices.push_back(b); mesh.indices.push_back(c);
            mesh.indices.push_back(a); mesh.indices.push_back(c); mesh.indices.push_back(d);
        }
    }
    return mesh;
}

// ---------- Sphere ----------
QVector3D Sphere::evaluate(float u, float v) const {
    return { std::sin(u) * std::cos(v),
             std::sin(u) * std::sin(v),
             std::cos(u) };
}
float Sphere::uMax() const { return kPi; }
float Sphere::vMax() const { return kTwoPi; }

// ---------- Torus ----------
QVector3D Torus::evaluate(float u, float v) const {
    const float cu = std::cos(u), su = std::sin(u);
    const float cv = std::cos(v), sv = std::sin(v);
    return { (R_ + r_ * cv) * cu,
             (R_ + r_ * cv) * su,
              r_ * sv };
}
float Torus::uMax() const { return kTwoPi; }
float Torus::vMax() const { return kTwoPi; }

// ---------- HelixTube ----------
// Center curve: c(u) = (R cos u, R sin u, p u)
// Frenet-style tube: surface(u, v) = c(u) + tube*(N(u) cos v + B(u) sin v)
QVector3D HelixTube::evaluate(float u, float v) const {
    const float cu = std::cos(u), su = std::sin(u);
    const QVector3D c(radius_ * cu, radius_ * su, pitch_ * u);
    QVector3D T(-radius_ * su, radius_ * cu, pitch_);
    T.normalize();
    // Stable normal: project z up onto plane perp to T
    QVector3D up(0.0f, 0.0f, 1.0f);
    QVector3D N = up - QVector3D::dotProduct(up, T) * T;
    if (N.lengthSquared() < 1e-6f) N = QVector3D(1, 0, 0);
    N.normalize();
    QVector3D B = QVector3D::crossProduct(T, N);
    return c + tube_ * (std::cos(v) * N + std::sin(v) * B);
}
float HelixTube::uMax() const { return turns_ * kTwoPi; }
float HelixTube::vMax() const { return kTwoPi; }

// ---------- Mobius ----------
QVector3D Mobius::evaluate(float u, float v) const {
    const float cu = std::cos(u), su = std::sin(u);
    const float h = 0.5f * u;
    const float ch = std::cos(h), sh = std::sin(h);
    return { (1.0f + v * ch) * cu,
             (1.0f + v * ch) * su,
              v * sh };
}
float Mobius::uMax() const { return kTwoPi; }

// ---------- Klein ----------
// Klein bottle: figure-8 immersion (compact, looks clean)
QVector3D Klein::evaluate(float u, float v) const {
    const float cu = std::cos(u), su = std::sin(u);
    const float cv = std::cos(v), sv = std::sin(v);
    const float r = 0.6f * (1.0f - 0.5f * cu);
    if (u < kPi) {
        return { 3.0f * cu * (1.0f + su) + r * cu * cv,
                 8.0f * su + r * su * cv,
                 r * sv };
    }
    return { 3.0f * cu * (1.0f + su) + r * std::cos(v + kPi),
             8.0f * su,
             r * sv };
}
float Klein::uMax() const { return kTwoPi; }
float Klein::vMax() const { return kTwoPi; }

// ---------- Saddle ----------
QVector3D Saddle::evaluate(float u, float v) const {
    return { u, v, u * v };
}

// ---------- Conical spiral ----------
// Spiral whose radius shrinks with height; surface = tube around the curve.
QVector3D ConicalSpiral::evaluate(float u, float v) const {
    const float radius = 1.0f - u / (3.0f * kTwoPi);
    const float cu = std::cos(u), su = std::sin(u);
    const QVector3D c(radius * cu, radius * su, 0.18f * u);
    // Approximate frame
    QVector3D T(-radius * su - cu * (1.0f / (3.0f * kTwoPi)),
                 radius * cu - su * (1.0f / (3.0f * kTwoPi)),
                 0.18f);
    T.normalize();
    QVector3D up(0.0f, 0.0f, 1.0f);
    QVector3D N = up - QVector3D::dotProduct(up, T) * T;
    if (N.lengthSquared() < 1e-6f) N = QVector3D(1, 0, 0);
    N.normalize();
    QVector3D B = QVector3D::crossProduct(T, N);
    const float tube = 0.08f * (0.5f + 0.5f * (1.0f - u / (3.0f * kTwoPi)));
    return c + tube * (std::cos(v) * N + std::sin(v) * B);
}
float ConicalSpiral::uMax() const { return 3.0f * kTwoPi; }
float ConicalSpiral::vMax() const { return kTwoPi; }

// ---------- Catalogue ----------
QVector<std::shared_ptr<Surface>> allSurfaces() {
    return {
        std::make_shared<Sphere>(),
        std::make_shared<Torus>(),
        std::make_shared<HelixTube>(),
        std::make_shared<ConicalSpiral>(),
        std::make_shared<Mobius>(),
        std::make_shared<Klein>(),
        std::make_shared<Saddle>(),
    };
}
