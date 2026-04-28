#pragma once

#include <QString>
#include <QVector>
#include <QVector3D>
#include <QColor>

// A small 3D "diagram" — a collection of primitives that illustrate the
// angles and distances behind a surface's parameterization. Rendered by
// DerivationView3D with a free orbit camera so the user can look at it
// from any direction.

namespace diag {

struct Point   { QVector3D p; QColor color; float size = 6.0f; QString label; };
struct Segment { QVector3D a; QVector3D b; QColor color; float width = 2.0f;
                 bool dashed = false; QString label; QVector3D labelAt; };

// An angle marker — a small arc of a circle in 3D.
//   centre: vertex of the angle
//   from:   first edge direction (will be normalized; arc starts from here)
//   to:     second edge direction (will be normalized; arc ends at here)
//   radius: arc radius
//   label:  text drawn near mid-arc
struct ArcAngle { QVector3D centre; QVector3D from; QVector3D to;
                  float radius; QColor color; QString label; };

// A circle / ellipse outline (rendered as a polyline).
//   centre: centre of the circle
//   axisU, axisV: two non-collinear vectors that span the plane of the circle.
//                  Their lengths set the ellipse semi-axes.
//   dashed: stroke style
struct Circle  { QVector3D centre; QVector3D axisU; QVector3D axisV;
                 QColor color; float width = 1.4f; bool dashed = false; };

// A flat plane patch (drawn as a translucent polygon) — useful to show
// e.g. the xy-plane.
struct Plane   { QVector3D centre; QVector3D axisU; QVector3D axisV;
                 QColor color; };

struct Scene {
    QString title;
    QVector<Plane>    planes;
    QVector<Circle>   circles;
    QVector<Segment>  segments;
    QVector<ArcAngle> arcs;
    QVector<Point>    points;
    float boundsRadius = 2.5f;  // for camera framing
};

Scene sceneFor(const QString& surfaceName);

}  // namespace diag
