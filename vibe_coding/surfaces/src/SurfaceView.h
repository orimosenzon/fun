#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>
#include <QMatrix4x4>
#include <QPoint>
#include <memory>

#include "Surface.h"

class SurfaceView : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
    Q_OBJECT
public:
    explicit SurfaceView(QWidget* parent = nullptr);
    ~SurfaceView() override;

    void setSurface(std::shared_ptr<Surface> s);
    void setAutoRotate(bool on);
    void resetView();

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void wheelEvent(QWheelEvent*) override;

private:
    void uploadMesh();
    void normalizeMesh(Mesh& mesh) const;

    std::shared_ptr<Surface> surface_;
    Mesh mesh_;

    QOpenGLShaderProgram program_;
    QOpenGLVertexArrayObject vao_;
    QOpenGLBuffer vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer ibo_{QOpenGLBuffer::IndexBuffer};
    int indexCount_ = 0;

    // Camera (orbit)
    float camYaw_ = 0.6f;
    float camPitch_ = 0.4f;
    float camDistance_ = 4.0f;
    QPoint lastMouse_;

    bool autoRotate_ = true;
    float autoAngle_ = 0.0f;
};
