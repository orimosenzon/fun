#pragma once

#include <QString>
#include <QVector3D>
#include <QVector>
#include <functional>
#include <memory>

struct Vertex {
    QVector3D position;
    QVector3D normal;
    float u;
    float v;
};

struct Mesh {
    QVector<Vertex> vertices;
    QVector<unsigned int> indices;
};

class Surface {
public:
    virtual ~Surface() = default;

    virtual QString name() const = 0;

    // The defining map: (u, v) in R^2 -> point in R^3
    virtual QVector3D evaluate(float u, float v) const = 0;

    // Parameter domain
    virtual float uMin() const = 0;
    virtual float uMax() const = 0;
    virtual float vMin() const = 0;
    virtual float vMax() const = 0;

    // Sampling resolution defaults; surfaces may override
    virtual int uSamples() const { return 96; }
    virtual int vSamples() const { return 96; }

    // Build a triangulated mesh from the parameterization. Normals are
    // estimated by finite differences of the parameterization.
    Mesh tessellate() const;
};

// f(u,v) = (sin u cos v, sin u sin v, cos u)
class Sphere : public Surface {
public:
    QString name() const override { return QStringLiteral("Sphere"); }
    QVector3D evaluate(float u, float v) const override;
    float uMin() const override { return 0.0f; }
    float uMax() const override;       // pi
    float vMin() const override { return 0.0f; }
    float vMax() const override;       // 2 pi
};

// Torus with major radius R, minor radius r
class Torus : public Surface {
public:
    explicit Torus(float R = 1.2f, float r = 0.45f) : R_(R), r_(r) {}
    QString name() const override { return QStringLiteral("Torus"); }
    QVector3D evaluate(float u, float v) const override;
    float uMin() const override { return 0.0f; }
    float uMax() const override;
    float vMin() const override { return 0.0f; }
    float vMax() const override;
private:
    float R_, r_;
};

// 3D helix as a tube: center curve is a helix, surface is its tubular
// neighborhood (u along the curve, v around the tube cross-section).
class HelixTube : public Surface {
public:
    explicit HelixTube(float radius = 1.0f, float pitch = 0.35f,
                       int turns = 4, float tube = 0.18f)
        : radius_(radius), pitch_(pitch), turns_(turns), tube_(tube) {}
    QString name() const override { return QStringLiteral("3D Helix (tube)"); }
    QVector3D evaluate(float u, float v) const override;
    float uMin() const override { return 0.0f; }
    float uMax() const override;
    float vMin() const override { return 0.0f; }
    float vMax() const override;
    int uSamples() const override { return 400; }
    int vSamples() const override { return 32; }
private:
    float radius_, pitch_;
    int turns_;
    float tube_;
};

// Möbius strip
class Mobius : public Surface {
public:
    QString name() const override { return QStringLiteral("Mobius strip"); }
    QVector3D evaluate(float u, float v) const override;
    float uMin() const override { return 0.0f; }
    float uMax() const override;
    float vMin() const override { return -0.4f; }
    float vMax() const override { return  0.4f; }
};

// Klein bottle (standard immersion in R^3)
class Klein : public Surface {
public:
    QString name() const override { return QStringLiteral("Klein bottle"); }
    QVector3D evaluate(float u, float v) const override;
    float uMin() const override { return 0.0f; }
    float uMax() const override;
    float vMin() const override { return 0.0f; }
    float vMax() const override;
};

// Hyperbolic paraboloid (saddle): f(u,v) = (u, v, u*v)
class Saddle : public Surface {
public:
    QString name() const override { return QStringLiteral("Saddle (z = u*v)"); }
    QVector3D evaluate(float u, float v) const override;
    float uMin() const override { return -1.0f; }
    float uMax() const override { return  1.0f; }
    float vMin() const override { return -1.0f; }
    float vMax() const override { return  1.0f; }
};

// Conical helix shell — combines a spiral with a vertical extrusion
class ConicalSpiral : public Surface {
public:
    QString name() const override { return QStringLiteral("Conical spiral"); }
    QVector3D evaluate(float u, float v) const override;
    float uMin() const override { return 0.0f; }
    float uMax() const override;
    float vMin() const override { return 0.0f; }
    float vMax() const override;
    int uSamples() const override { return 300; }
    int vSamples() const override { return 24; }
};

// Catenoid: surface of revolution of the catenary cosh(v); minimal surface
class Catenoid : public Surface {
public:
    QString name() const override { return QStringLiteral("Catenoid"); }
    QVector3D evaluate(float u, float v) const override;
    float uMin() const override { return 0.0f; }
    float uMax() const override;   // 2π
    float vMin() const override { return -2.0f; }
    float vMax() const override { return  2.0f; }
};

// Helicoid: rotating half-line that also rises — S(u,v) = (v cos u, v sin u, u/π)
class Helicoid : public Surface {
public:
    QString name() const override { return QStringLiteral("Helicoid"); }
    QVector3D evaluate(float u, float v) const override;
    float uMin() const override { return 0.0f; }
    float uMax() const override;   // 4π — two full turns
    float vMin() const override { return -1.0f; }
    float vMax() const override { return  1.0f; }
};

// Enneper surface: polynomial minimal surface with self-intersection
class Enneper : public Surface {
public:
    QString name() const override { return QStringLiteral("Enneper surface"); }
    QVector3D evaluate(float u, float v) const override;
    float uMin() const override { return -1.5f; }
    float uMax() const override { return  1.5f; }
    float vMin() const override { return -1.5f; }
    float vMax() const override { return  1.5f; }
};

// Catalogue of all available surfaces
QVector<std::shared_ptr<Surface>> allSurfaces();
