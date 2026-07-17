#pragma once
#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShaderProgram>
#include <QElapsedTimer>
#include <QTimer>
#include <QVector3D>
#include <QMatrix4x4>
#include "Mesh.h"

struct Prop {
    QVector3D position;
    float scale;
    float rotationDeg;
};

class GameView : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
    Q_OBJECT
public:
    explicit GameView(QWidget* parent = nullptr);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;

private slots:
    void tick();

private:
    void buildScene();
    void drawObject(Mesh& mesh, const QMatrix4x4& model, const QVector3D& color);
    void updatePhysics(float dt);

    QOpenGLShaderProgram m_program;
    Mesh m_cube, m_cylinder, m_cone, m_plane, m_ring, m_dash;

    QMatrix4x4 m_proj;
    QTimer m_timer;
    QElapsedTimer m_clock;

    // Track geometry.
    static constexpr float kInnerRadius = 22.0f;
    static constexpr float kOuterRadius = 32.0f;
    static constexpr float kTrackCenterRadius = (kInnerRadius + kOuterRadius) / 2.0f;

    // Car state.
    QVector3D m_carPos{kTrackCenterRadius, 0.0f, 0.0f};
    float m_heading = float(M_PI); // tangent to the track at the start position
    float m_speed = 0.0f;
    bool m_keyForward = false, m_keyBackward = false, m_keyLeft = false, m_keyRight = false;

    int m_lapCount = 0;
    float m_prevAngle = float(M_PI) / 2.0f; // matches the initial car angle

    std::vector<Prop> m_trees;
    std::vector<Prop> m_houses;
};
