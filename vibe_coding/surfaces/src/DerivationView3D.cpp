#include "DerivationView3D.h"

#include <QMouseEvent>
#include <QWheelEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QFont>
#include <QFontMetricsF>
#include <QtMath>
#include <cmath>

namespace {

constexpr const char* kVS = R"GLSL(
#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_color;
uniform mat4 u_mvp;
out vec3 v_color;
void main() {
    v_color = a_color;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
)GLSL";

constexpr const char* kFS = R"GLSL(
#version 330 core
in vec3 v_color;
out vec4 fragColor;
void main() {
    fragColor = vec4(v_color, 1.0);
}
)GLSL";

void appendVertex(QVector<float>& v, const QVector3D& p, const QColor& c) {
    v << p.x() << p.y() << p.z()
      << float(c.redF()) << float(c.greenF()) << float(c.blueF());
}

}  // namespace

DerivationView3D::DerivationView3D(QWidget* parent) : QOpenGLWidget(parent) {
    QSurfaceFormat fmt;
    fmt.setVersion(3, 3);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    fmt.setSamples(4);
    fmt.setDepthBufferSize(24);
    setFormat(fmt);
    setMinimumHeight(280);
}

DerivationView3D::~DerivationView3D() {
    if (glReady_) {
        makeCurrent();
        vbo_.destroy();
        vao_.destroy();
        doneCurrent();
    }
}

void DerivationView3D::setScene(const diag::Scene& s) {
    scene_ = s;
    update();
}

void DerivationView3D::resetView() {
    yaw_ = 0.7f;
    pitch_ = 0.45f;
    distance_ = 5.5f;
    update();
}

void DerivationView3D::initializeGL() {
    initializeOpenGLFunctions();
    glReady_ = true;
    glClearColor(0.105f, 0.110f, 0.125f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    lineProgram_.addShaderFromSourceCode(QOpenGLShader::Vertex, kVS);
    lineProgram_.addShaderFromSourceCode(QOpenGLShader::Fragment, kFS);
    lineProgram_.link();

    vao_.create();
    vbo_.create();
}

void DerivationView3D::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}

QMatrix4x4 DerivationView3D::projection() const {
    QMatrix4x4 p;
    const float aspect = float(width()) / float(std::max(1, height()));
    p.perspective(38.0f, aspect, 0.05f, 100.0f);
    return p;
}

QVector3D DerivationView3D::eyePos() const {
    const float cp = std::cos(pitch_), sp = std::sin(pitch_);
    const float cy = std::cos(yaw_),   sy = std::sin(yaw_);
    return { distance_ * cp * sy,
             distance_ * cp * cy,
             distance_ * sp };
}

QMatrix4x4 DerivationView3D::viewMatrix() const {
    QMatrix4x4 v;
    v.lookAt(eyePos(), QVector3D(0, 0, 0), QVector3D(0, 0, 1));
    return v;
}

QPointF DerivationView3D::projectToScreen(const QVector3D& world,
                                          const QMatrix4x4& mvp) const {
    const QVector4D p = mvp * QVector4D(world, 1.0f);
    if (p.w() <= 0.0001f) return { -10000, -10000 };
    const float ndcX = p.x() / p.w();
    const float ndcY = p.y() / p.w();
    return { (ndcX * 0.5f + 0.5f) * width(),
             (1.0f - (ndcY * 0.5f + 0.5f)) * height() };
}

void DerivationView3D::drawLines(const QVector<float>& v,
                                 const QMatrix4x4& mvp) {
    if (v.isEmpty()) return;
    vao_.bind();
    vbo_.bind();
    vbo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);
    vbo_.allocate(v.constData(), v.size() * sizeof(float));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          (void*)(3 * sizeof(float)));
    lineProgram_.bind();
    lineProgram_.setUniformValue("u_mvp", mvp);
    glDrawArrays(GL_LINES, 0, v.size() / 6);
    lineProgram_.release();
    vbo_.release();
    vao_.release();
}

void DerivationView3D::drawPoints(const QVector<float>& v,
                                  const QMatrix4x4& mvp) {
    if (v.isEmpty()) return;
    vao_.bind();
    vbo_.bind();
    vbo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);
    vbo_.allocate(v.constData(), v.size() * sizeof(float));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          (void*)(3 * sizeof(float)));
    lineProgram_.bind();
    lineProgram_.setUniformValue("u_mvp", mvp);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glPointSize(8.0f);
    glDrawArrays(GL_POINTS, 0, v.size() / 6);
    lineProgram_.release();
    vbo_.release();
    vao_.release();
}

void DerivationView3D::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (scene_.segments.isEmpty() && scene_.circles.isEmpty()
        && scene_.arcs.isEmpty()    && scene_.points.isEmpty()) return;

    const QMatrix4x4 proj = projection();
    const QMatrix4x4 view = viewMatrix();
    const QMatrix4x4 mvp  = proj * view;

    // ── Planes (translucent quads, drawn via two tri pairs as line strips for now) ──
    // Render planes as semi-transparent fills using GL_TRIANGLES, in their own batch.
    {
        QVector<float> tri;
        for (const auto& pl : scene_.planes) {
            const QVector3D a = pl.centre - 0.5f*pl.axisU - 0.5f*pl.axisV;
            const QVector3D b = pl.centre + 0.5f*pl.axisU - 0.5f*pl.axisV;
            const QVector3D c = pl.centre + 0.5f*pl.axisU + 0.5f*pl.axisV;
            const QVector3D d = pl.centre - 0.5f*pl.axisU + 0.5f*pl.axisV;
            const QColor col = pl.color;
            const float r = col.redF(), g = col.greenF(), bl = col.blueF();
            const float al = float(col.alphaF());  // unused; using opaque-ish
            (void)al;
            // emit 6 vertices (two triangles)
            for (auto v : { a, b, c, a, c, d }) {
                tri << v.x() << v.y() << v.z() << r << g << bl;
            }
        }
        if (!tri.isEmpty()) {
            vao_.bind();
            vbo_.bind();
            vbo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);
            vbo_.allocate(tri.constData(), tri.size() * sizeof(float));
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)0);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float),
                                  (void*)(3*sizeof(float)));
            lineProgram_.bind();
            lineProgram_.setUniformValue("u_mvp", mvp);
            glDepthMask(GL_FALSE);
            glDrawArrays(GL_TRIANGLES, 0, tri.size()/6);
            glDepthMask(GL_TRUE);
            lineProgram_.release();
            vbo_.release();
            vao_.release();
        }
    }

    // Build line vertex buffer from segments, circles (polylines), arcs
    QVector<float> lines;

    // Segments
    for (const auto& s : scene_.segments) {
        if (s.dashed) {
            // emit ~12 dashes
            const int K = 16;
            for (int i = 0; i < K; ++i) {
                float t1 = (i + 0.0f) / K;
                float t2 = (i + 0.55f) / K;
                if (t2 > 1.0f) t2 = 1.0f;
                if ((i & 1) == 0) {
                    QVector3D p1 = s.a + t1 * (s.b - s.a);
                    QVector3D p2 = s.a + t2 * (s.b - s.a);
                    appendVertex(lines, p1, s.color);
                    appendVertex(lines, p2, s.color);
                }
            }
        } else {
            appendVertex(lines, s.a, s.color);
            appendVertex(lines, s.b, s.color);
        }
    }

    // Circles (polylines)
    for (const auto& c : scene_.circles) {
        const int K = 96;
        for (int i = 0; i < K; ++i) {
            const float t1 = 2.0f * float(M_PI) * i / K;
            const float t2 = 2.0f * float(M_PI) * (i + 1) / K;
            if (c.dashed && (i & 1)) continue;
            QVector3D p1 = c.centre + std::cos(t1)*c.axisU + std::sin(t1)*c.axisV;
            QVector3D p2 = c.centre + std::cos(t2)*c.axisU + std::sin(t2)*c.axisV;
            appendVertex(lines, p1, c.color);
            appendVertex(lines, p2, c.color);
        }
    }

    // Arcs
    for (const auto& a : scene_.arcs) {
        QVector3D u = a.from.normalized();
        QVector3D w = a.to.normalized();
        // Build orthonormal frame (u, perp) within the plane spanned by u and w.
        QVector3D normal = QVector3D::crossProduct(u, w);
        if (normal.lengthSquared() < 1e-8f) continue;
        normal.normalize();
        QVector3D perp = QVector3D::crossProduct(normal, u).normalized();
        const float angle = std::acos(std::clamp(QVector3D::dotProduct(u, w), -1.0f, 1.0f));
        const int K = 32;
        for (int i = 0; i < K; ++i) {
            const float t1 = angle * i / K;
            const float t2 = angle * (i + 1) / K;
            QVector3D p1 = a.centre + a.radius * (std::cos(t1)*u + std::sin(t1)*perp);
            QVector3D p2 = a.centre + a.radius * (std::cos(t2)*u + std::sin(t2)*perp);
            appendVertex(lines, p1, a.color);
            appendVertex(lines, p2, a.color);
        }
    }

    // Set GL line width hint (driver-dependent)
    glLineWidth(1.6f);
    drawLines(lines, mvp);

    // Points
    QVector<float> pts;
    for (const auto& p : scene_.points) appendVertex(pts, p.p, p.color);
    drawPoints(pts, mvp);
}

void DerivationView3D::paintEvent(QPaintEvent* e) {
    QOpenGLWidget::paintEvent(e);

    // Now overlay text labels using QPainter on the widget's surface.
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::TextAntialiasing, true);

    QFont f = painter.font();
    f.setPointSizeF(10.5);
    f.setBold(true);
    painter.setFont(f);

    const QMatrix4x4 mvp = projection() * viewMatrix();
    QFontMetricsF fm(f);

    auto drawLabel = [&](const QVector3D& world, const QString& text, const QColor& col) {
        if (text.isEmpty()) return;
        QPointF s = projectToScreen(world, mvp);
        if (s.x() < -100 || s.x() > width() + 100) return;
        // bg pill for legibility
        const qreal pad = 3.0;
        const QRectF tr = fm.tightBoundingRect(text).translated(s.x() + 6, s.y() - 4);
        QRectF bg(tr.left() - pad, tr.top() - pad,
                  tr.width() + 2*pad, tr.height() + 2*pad);
        painter.setBrush(QColor(15, 16, 22, 200));
        painter.setPen(Qt::NoPen);
        painter.drawRoundedRect(bg, 3, 3);
        painter.setPen(col);
        painter.drawText(QPointF(s.x() + 6, s.y() - 4 + tr.height()/2 + fm.ascent()/2 - 1),
                         text);
    };

    for (const auto& seg : scene_.segments) {
        if (!seg.label.isEmpty()) drawLabel(seg.labelAt, seg.label, seg.color);
    }
    for (const auto& a : scene_.arcs) {
        if (a.label.isEmpty()) continue;
        QVector3D u = a.from.normalized();
        QVector3D w = a.to.normalized();
        QVector3D normal = QVector3D::crossProduct(u, w);
        if (normal.lengthSquared() < 1e-8f) continue;
        normal.normalize();
        QVector3D perp = QVector3D::crossProduct(normal, u).normalized();
        const float angle = std::acos(std::clamp(QVector3D::dotProduct(u, w), -1.0f, 1.0f));
        const float t = angle * 0.5f;
        QVector3D mid = a.centre + (a.radius * 1.18f) *
                        (std::cos(t)*u + std::sin(t)*perp);
        drawLabel(mid, a.label, a.color);
    }
    for (const auto& pt : scene_.points) {
        if (!pt.label.isEmpty()) drawLabel(pt.p, pt.label, pt.color);
    }

    // Title bar
    if (!scene_.title.isEmpty()) {
        QFont tf = f;
        tf.setPointSizeF(10.5);
        tf.setBold(false);
        painter.setFont(tf);
        painter.setPen(QColor(200, 205, 220));
        painter.drawText(QRect(0, 4, width(), 22), Qt::AlignHCenter, scene_.title);
    }

    // Hint text in corner
    QFont hf = f;
    hf.setPointSizeF(8.5);
    hf.setBold(false);
    painter.setFont(hf);
    painter.setPen(QColor(140, 145, 160));
    painter.drawText(QRect(8, height() - 22, width() - 16, 18),
                     Qt::AlignLeft,
                     QStringLiteral("גרור — סיבוב   ·   גלגל — זום"));
}

void DerivationView3D::mousePressEvent(QMouseEvent* e) { lastMouse_ = e->pos(); }

void DerivationView3D::mouseMoveEvent(QMouseEvent* e) {
    const QPoint d = e->pos() - lastMouse_;
    lastMouse_ = e->pos();
    if (e->buttons() & Qt::LeftButton) {
        yaw_   += d.x() * 0.01f;
        pitch_ += d.y() * 0.01f;
        const float lim = 1.55f;
        if (pitch_ >  lim) pitch_ =  lim;
        if (pitch_ < -lim) pitch_ = -lim;
        update();
    }
}

void DerivationView3D::wheelEvent(QWheelEvent* e) {
    const float steps = e->angleDelta().y() / 120.0f;
    distance_ *= std::pow(0.9f, steps);
    if (distance_ < 1.5f)  distance_ = 1.5f;
    if (distance_ > 30.0f) distance_ = 30.0f;
    update();
}
