#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>
#include <QMatrix4x4>
#include <QPoint>

#include "DiagramScene.h"

class DerivationView3D : public QOpenGLWidget,
                         protected QOpenGLFunctions_3_3_Core {
    Q_OBJECT
public:
    explicit DerivationView3D(QWidget* parent = nullptr);
    ~DerivationView3D() override;

    void setScene(const diag::Scene& s);
    void resetView();

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void paintEvent(QPaintEvent* e) override;

    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void wheelEvent(QWheelEvent*) override;

private:
    QMatrix4x4 projection() const;
    QMatrix4x4 viewMatrix() const;
    QVector3D  eyePos() const;
    QPointF projectToScreen(const QVector3D& world,
                            const QMatrix4x4& mvp) const;

    void drawLines(const QVector<float>& vertices,
                   const QMatrix4x4& mvp);
    void drawPoints(const QVector<float>& vertices,
                    const QMatrix4x4& mvp);

    diag::Scene scene_;

    QOpenGLShaderProgram lineProgram_;
    QOpenGLVertexArrayObject vao_;
    QOpenGLBuffer vbo_{QOpenGLBuffer::VertexBuffer};
    bool glReady_ = false;

    // Orbit camera
    float yaw_ = 0.7f;
    float pitch_ = 0.45f;
    float distance_ = 5.5f;
    QPoint lastMouse_;
};
