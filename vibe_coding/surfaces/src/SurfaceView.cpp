#include "SurfaceView.h"

#include <QMouseEvent>
#include <QWheelEvent>
#include <QTimer>
#include <QFile>
#include <QDebug>
#include <cmath>

SurfaceView::SurfaceView(QWidget* parent) : QOpenGLWidget(parent) {
    QSurfaceFormat fmt;
    fmt.setVersion(3, 3);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    fmt.setSamples(4);
    fmt.setDepthBufferSize(24);
    setFormat(fmt);

    auto* timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, [this] {
        if (autoRotate_) {
            autoAngle_ += 0.012f;
            update();
        }
    });
    timer->start(16);
}

SurfaceView::~SurfaceView() {
    makeCurrent();
    vbo_.destroy();
    ibo_.destroy();
    vao_.destroy();
    doneCurrent();
}

void SurfaceView::setSurface(std::shared_ptr<Surface> s) {
    surface_ = std::move(s);
    if (!surface_) return;
    if (context()) {
        makeCurrent();
        uploadMesh();
        doneCurrent();
        update();
    }
}

void SurfaceView::setAutoRotate(bool on) { autoRotate_ = on; }

void SurfaceView::resetView() {
    camYaw_ = 0.6f;
    camPitch_ = 0.4f;
    camDistance_ = 4.0f;
    autoAngle_ = 0.0f;
    update();
}

void SurfaceView::initializeGL() {
    initializeOpenGLFunctions();
    glClearColor(0.07f, 0.08f, 0.10f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    if (!program_.addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/surface.vert")) {
        qWarning() << "vertex:" << program_.log();
    }
    if (!program_.addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/surface.frag")) {
        qWarning() << "fragment:" << program_.log();
    }
    if (!program_.link()) {
        qWarning() << "link:" << program_.log();
    }

    vao_.create();
    vbo_.create();
    ibo_.create();

    if (surface_) uploadMesh();
}

void SurfaceView::normalizeMesh(Mesh& mesh) const {
    if (mesh.vertices.isEmpty()) return;
    QVector3D mn = mesh.vertices.first().position;
    QVector3D mx = mn;
    for (const auto& v : mesh.vertices) {
        mn.setX(std::min(mn.x(), v.position.x()));
        mn.setY(std::min(mn.y(), v.position.y()));
        mn.setZ(std::min(mn.z(), v.position.z()));
        mx.setX(std::max(mx.x(), v.position.x()));
        mx.setY(std::max(mx.y(), v.position.y()));
        mx.setZ(std::max(mx.z(), v.position.z()));
    }
    QVector3D center = 0.5f * (mn + mx);
    QVector3D size = mx - mn;
    float maxDim = std::max({size.x(), size.y(), size.z()});
    float scale = (maxDim > 1e-6f) ? (2.0f / maxDim) : 1.0f;
    for (auto& v : mesh.vertices) {
        v.position = (v.position - center) * scale;
    }
}

void SurfaceView::uploadMesh() {
    mesh_ = surface_->tessellate();
    normalizeMesh(mesh_);

    QVector<float> packed;
    packed.reserve(mesh_.vertices.size() * 8);
    for (const auto& v : mesh_.vertices) {
        packed << v.position.x() << v.position.y() << v.position.z()
               << v.normal.x()   << v.normal.y()   << v.normal.z()
               << v.u            << v.v;
    }

    vao_.bind();
    vbo_.bind();
    vbo_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    vbo_.allocate(packed.constData(), packed.size() * sizeof(float));

    ibo_.bind();
    ibo_.setUsagePattern(QOpenGLBuffer::StaticDraw);
    ibo_.allocate(mesh_.indices.constData(),
                  mesh_.indices.size() * sizeof(unsigned int));

    const int stride = 8 * sizeof(float);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));

    indexCount_ = mesh_.indices.size();

    // Release VAO first so it remembers the IBO binding;
    // releasing IBO while the VAO is bound would unbind it from the VAO.
    vao_.release();
    vbo_.release();
    ibo_.release();
}

void SurfaceView::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}

void SurfaceView::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (!indexCount_ || !surface_) return;

    const float aspect = float(width()) / float(std::max(1, height()));

    QMatrix4x4 proj;
    proj.perspective(45.0f, aspect, 0.05f, 100.0f);

    QMatrix4x4 view;
    const float yaw = camYaw_ + autoAngle_;
    const float cy = std::cos(yaw), sy = std::sin(yaw);
    const float cp = std::cos(camPitch_), sp = std::sin(camPitch_);
    QVector3D eye(camDistance_ * cp * sy,
                  camDistance_ * sp,
                  camDistance_ * cp * cy);
    view.lookAt(eye, QVector3D(0, 0, 0), QVector3D(0, 1, 0));

    QMatrix4x4 model;
    // rotate so Z points up in math but Y is up in screen
    model.rotate(-90.0f, 1.0f, 0.0f, 0.0f);

    QMatrix3x3 nrm = model.normalMatrix();

    program_.bind();
    program_.setUniformValue("u_proj", proj);
    program_.setUniformValue("u_view", view);
    program_.setUniformValue("u_model", model);
    program_.setUniformValue("u_normalMatrix", nrm);
    program_.setUniformValue("u_camPos", eye);
    program_.setUniformValue("u_lightDir", QVector3D(-0.4f, -0.8f, -0.5f).normalized());
    program_.setUniformValue("u_baseColor", QVector3D(0.35f, 0.62f, 0.95f));
    program_.setUniformValue("u_uMin", surface_->uMin());
    program_.setUniformValue("u_uMax", surface_->uMax());
    program_.setUniformValue("u_vMin", surface_->vMin());
    program_.setUniformValue("u_vMax", surface_->vMax());
    program_.setUniformValue("u_doubleSided", true);

    vao_.bind();
    glDrawElements(GL_TRIANGLES, indexCount_, GL_UNSIGNED_INT, nullptr);
    vao_.release();
    program_.release();
}

void SurfaceView::mousePressEvent(QMouseEvent* e) {
    lastMouse_ = e->pos();
}

void SurfaceView::mouseMoveEvent(QMouseEvent* e) {
    const QPoint d = e->pos() - lastMouse_;
    lastMouse_ = e->pos();
    if (e->buttons() & Qt::LeftButton) {
        camYaw_   += d.x() * 0.01f;
        camPitch_ += d.y() * 0.01f;
        const float lim = 1.55f;
        if (camPitch_ >  lim) camPitch_ =  lim;
        if (camPitch_ < -lim) camPitch_ = -lim;
        update();
    }
}

void SurfaceView::wheelEvent(QWheelEvent* e) {
    const float steps = e->angleDelta().y() / 120.0f;
    camDistance_ *= std::pow(0.9f, steps);
    if (camDistance_ < 1.5f) camDistance_ = 1.5f;
    if (camDistance_ > 25.0f) camDistance_ = 25.0f;
    update();
}
