#include "GameView.h"
#include <QKeyEvent>
#include <QPainter>
#include <QFont>
#include <QRandomGenerator>
#include <QtMath>
#include <cmath>
#include <algorithm>

static const char* kVertexShader = R"GLSL(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat3 uNormalMatrix;

out vec3 vNormal;
out vec3 vWorldPos;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal = normalize(uNormalMatrix * aNormal);
    gl_Position = uProj * uView * worldPos;
}
)GLSL";

static const char* kFragmentShader = R"GLSL(
#version 330 core
in vec3 vNormal;
in vec3 vWorldPos;
out vec4 FragColor;

uniform vec3 uColor;
uniform vec3 uLightDir;
uniform vec3 uViewPos;
uniform vec3 uFogColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(-uLightDir);
    float diff = max(dot(N, L), 0.0);

    vec3 V = normalize(uViewPos - vWorldPos);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 24.0) * 0.15;

    vec3 lit = uColor * (0.38 + 0.62 * diff) + vec3(spec);

    float dist = length(uViewPos - vWorldPos);
    float fogFactor = clamp((dist - 40.0) / 160.0, 0.0, 0.85);
    vec3 finalColor = mix(lit, uFogColor, fogFactor);

    FragColor = vec4(finalColor, 1.0);
}
)GLSL";

GameView::GameView(QWidget* parent) : QOpenGLWidget(parent) {
    setFocusPolicy(Qt::StrongFocus);
    QSurfaceFormat fmt;
    fmt.setVersion(3, 3);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    fmt.setDepthBufferSize(24);
    fmt.setSamples(4);
    setFormat(fmt);
}

void GameView::initializeGL() {
    initializeOpenGLFunctions();
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glClearColor(0.53f, 0.72f, 0.90f, 1.0f);

    m_program.addShaderFromSourceCode(QOpenGLShader::Vertex, kVertexShader);
    m_program.addShaderFromSourceCode(QOpenGLShader::Fragment, kFragmentShader);
    m_program.link();

    std::vector<Vertex> v; std::vector<unsigned int> idx;
    MeshGen::cube(v, idx); m_cube.init(this, v, idx);

    v.clear(); idx.clear();
    MeshGen::cylinder(v, idx, 20); m_cylinder.init(this, v, idx);

    v.clear(); idx.clear();
    MeshGen::cone(v, idx, 16); m_cone.init(this, v, idx);

    v.clear(); idx.clear();
    MeshGen::plane(v, idx); m_plane.init(this, v, idx);

    v.clear(); idx.clear();
    MeshGen::ring(v, idx, kInnerRadius, kOuterRadius, 0.02f, 96); m_ring.init(this, v, idx);

    v.clear(); idx.clear();
    MeshGen::cube(v, idx); m_dash.init(this, v, idx); // reuse cube geometry for lane dashes

    buildScene();

    m_clock.start();
    connect(&m_timer, &QTimer::timeout, this, &GameView::tick);
    m_timer.start(16);
}

void GameView::buildScene() {
    QRandomGenerator rng(1337); // fixed seed: same scenic layout every run

    // Trees scattered outside the track, and a few in the infield.
    for (int i = 0; i < 90; ++i) {
        float angle = rng.bounded(1000) / 1000.0f * 2.0f * float(M_PI);
        float radius = kOuterRadius + 4.0f + rng.bounded(1000) / 1000.0f * 30.0f;
        Prop t;
        t.position = QVector3D(std::cos(angle) * radius, 0.0f, std::sin(angle) * radius);
        t.scale = 1.4f + rng.bounded(1000) / 1000.0f * 1.6f;
        t.rotationDeg = rng.bounded(360);
        m_trees.push_back(t);
    }
    for (int i = 0; i < 14; ++i) {
        float angle = rng.bounded(1000) / 1000.0f * 2.0f * float(M_PI);
        float radius = rng.bounded(1000) / 1000.0f * (kInnerRadius - 5.0f);
        Prop t;
        t.position = QVector3D(std::cos(angle) * radius, 0.0f, std::sin(angle) * radius);
        t.scale = 1.2f + rng.bounded(1000) / 1000.0f * 1.4f;
        t.rotationDeg = rng.bounded(360);
        m_trees.push_back(t);
    }

    // Houses further out, spaced around the track.
    int houseCount = 10;
    for (int i = 0; i < houseCount; ++i) {
        float angle = (2.0f * float(M_PI) * i) / houseCount + 0.3f;
        float radius = kOuterRadius + 16.0f + rng.bounded(1000) / 1000.0f * 8.0f;
        Prop h;
        h.position = QVector3D(std::cos(angle) * radius, 0.0f, std::sin(angle) * radius);
        h.scale = 3.0f + rng.bounded(1000) / 1000.0f * 1.5f;
        h.rotationDeg = rng.bounded(360);
        m_houses.push_back(h);
    }
}

void GameView::resizeGL(int w, int h) {
    m_proj.setToIdentity();
    float aspect = h > 0 ? float(w) / float(h) : 1.0f;
    m_proj.perspective(60.0f, aspect, 0.1f, 500.0f);
}

void GameView::keyPressEvent(QKeyEvent* event) {
    switch (event->key()) {
        case Qt::Key_Up: case Qt::Key_W: m_keyForward = true; break;
        case Qt::Key_Down: case Qt::Key_S: m_keyBackward = true; break;
        case Qt::Key_Left: case Qt::Key_A: m_keyLeft = true; break;
        case Qt::Key_Right: case Qt::Key_D: m_keyRight = true; break;
        case Qt::Key_Escape: window()->close(); break;
        default: QOpenGLWidget::keyPressEvent(event);
    }
}

void GameView::keyReleaseEvent(QKeyEvent* event) {
    switch (event->key()) {
        case Qt::Key_Up: case Qt::Key_W: m_keyForward = false; break;
        case Qt::Key_Down: case Qt::Key_S: m_keyBackward = false; break;
        case Qt::Key_Left: case Qt::Key_A: m_keyLeft = false; break;
        case Qt::Key_Right: case Qt::Key_D: m_keyRight = false; break;
        default: QOpenGLWidget::keyReleaseEvent(event);
    }
}

void GameView::tick() {
    float dt = m_clock.restart() / 1000.0f;
    dt = std::min(dt, 0.05f);
    updatePhysics(dt);
    update();
}

void GameView::updatePhysics(float dt) {
    const float accel = 16.0f;
    const float brakeAccel = 22.0f;
    const float friction = 9.0f;
    const float maxSpeed = 26.0f;
    const float turnRate = 2.4f; // rad/s at full speed

    if (m_keyForward) {
        m_speed += accel * dt;
    } else if (m_keyBackward) {
        m_speed -= brakeAccel * dt;
    } else {
        if (m_speed > 0.0f) m_speed = std::max(0.0f, m_speed - friction * dt);
        else m_speed = std::min(0.0f, m_speed + friction * dt);
    }
    m_speed = std::clamp(m_speed, -maxSpeed * 0.45f, maxSpeed);

    float turnInput = 0.0f;
    if (m_keyLeft) turnInput += 1.0f;
    if (m_keyRight) turnInput -= 1.0f;

    float speedFactor = std::clamp(std::fabs(m_speed) / maxSpeed, 0.2f, 1.0f);
    float dir = m_speed >= 0.0f ? 1.0f : -1.0f;
    m_heading += turnInput * turnRate * speedFactor * dir * dt;

    m_carPos += QVector3D(std::sin(m_heading), 0.0f, std::cos(m_heading)) * m_speed * dt;

    // Lap detection: count wraps of the polar angle while roughly on the track.
    float angle = std::atan2(m_carPos.x(), m_carPos.z());
    if (angle < 0.0f) angle += 2.0f * float(M_PI);
    float distFromCenter = std::hypot(m_carPos.x(), m_carPos.z());
    bool onTrack = distFromCenter > kInnerRadius - 3.0f && distFromCenter < kOuterRadius + 3.0f;
    if (onTrack && m_prevAngle > 5.5f && angle < 0.8f) {
        m_lapCount++;
    }
    m_prevAngle = angle;
}

void GameView::drawObject(Mesh& mesh, const QMatrix4x4& model, const QVector3D& color) {
    m_program.setUniformValue("uModel", model);
    m_program.setUniformValue("uNormalMatrix", model.normalMatrix());
    m_program.setUniformValue("uColor", color);
    mesh.draw(this);
}

void GameView::paintGL() {
    QPainter painter(this);
    painter.beginNativePainting();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    m_program.bind();

    QVector3D forward(std::sin(m_heading), 0.0f, std::cos(m_heading));
    QVector3D camPos = m_carPos - forward * 9.0f + QVector3D(0.0f, 4.5f, 0.0f);
    QVector3D lookAt = m_carPos + forward * 4.0f + QVector3D(0.0f, 1.0f, 0.0f);

    QMatrix4x4 view;
    view.lookAt(camPos, lookAt, QVector3D(0, 1, 0));

    QVector3D lightDir(-0.4f, -1.0f, -0.35f);
    QVector3D fogColor(0.53f, 0.72f, 0.90f);

    m_program.setUniformValue("uView", view);
    m_program.setUniformValue("uProj", m_proj);
    m_program.setUniformValue("uLightDir", lightDir);
    m_program.setUniformValue("uViewPos", camPos);
    m_program.setUniformValue("uFogColor", fogColor);

    // Ground.
    QMatrix4x4 groundModel;
    groundModel.scale(400.0f, 1.0f, 400.0f);
    drawObject(m_plane, groundModel, QVector3D(0.35f, 0.62f, 0.30f));

    // Track surface.
    QMatrix4x4 identity;
    drawObject(m_ring, identity, QVector3D(0.24f, 0.24f, 0.26f));

    // Dashed centerline.
    int dashCount = 48;
    for (int i = 0; i < dashCount; ++i) {
        if (i % 2 == 0) continue;
        float a = 2.0f * float(M_PI) * i / dashCount;
        QMatrix4x4 m;
        m.translate(std::cos(a) * kTrackCenterRadius, 0.03f, std::sin(a) * kTrackCenterRadius);
        m.rotate(-qRadiansToDegrees(a) + 90.0f, 0, 1, 0);
        m.scale(0.4f, 0.02f, 2.0f);
        drawObject(m_dash, m, QVector3D(0.95f, 0.95f, 0.9f));
    }

    // Trees: trunk (cylinder) + foliage (cone).
    for (const Prop& t : m_trees) {
        float trunkH = 1.6f * t.scale, trunkR = 0.18f * t.scale;
        QMatrix4x4 trunk;
        trunk.translate(t.position.x(), 0.0f, t.position.z());
        trunk.rotate(t.rotationDeg, 0, 1, 0);
        trunk.scale(trunkR, trunkH, trunkR);
        drawObject(m_cylinder, trunk, QVector3D(0.40f, 0.26f, 0.13f));

        float foliageH = 3.2f * t.scale, foliageR = 1.3f * t.scale;
        QMatrix4x4 foliage;
        foliage.translate(t.position.x(), trunkH, t.position.z());
        foliage.rotate(t.rotationDeg, 0, 1, 0);
        foliage.scale(foliageR, foliageH, foliageR);
        drawObject(m_cone, foliage, QVector3D(0.16f, 0.45f, 0.18f));
    }

    // Houses: box body + pyramid roof.
    for (const Prop& h : m_houses) {
        float bodyW = 3.2f * h.scale * 0.5f, bodyH = 2.4f * h.scale * 0.5f, bodyD = 3.2f * h.scale * 0.5f;
        QMatrix4x4 body;
        body.translate(h.position.x(), bodyH / 2.0f, h.position.z());
        body.rotate(h.rotationDeg, 0, 1, 0);
        body.scale(bodyW, bodyH, bodyD);
        drawObject(m_cube, body, QVector3D(0.85f, 0.78f, 0.65f));

        float roofH = 1.6f * h.scale * 0.5f, roofR = 2.6f * h.scale * 0.5f;
        QMatrix4x4 roof;
        roof.translate(h.position.x(), bodyH, h.position.z());
        roof.rotate(h.rotationDeg + 45.0f, 0, 1, 0);
        roof.scale(roofR, roofH, roofR);
        drawObject(m_cone, roof, QVector3D(0.55f, 0.18f, 0.15f));
    }

    // Car: body, cabin, four wheels.
    QMatrix4x4 carBase;
    carBase.translate(m_carPos.x(), 0.0f, m_carPos.z());
    carBase.rotate(qRadiansToDegrees(m_heading), 0, 1, 0);

    QMatrix4x4 body = carBase;
    body.translate(0.0f, 0.55f, 0.0f);
    body.scale(1.7f, 0.7f, 3.4f);
    drawObject(m_cube, body, QVector3D(0.82f, 0.10f, 0.10f));

    QMatrix4x4 cabin = carBase;
    cabin.translate(0.0f, 1.05f, -0.3f);
    cabin.scale(1.3f, 0.55f, 1.6f);
    drawObject(m_cube, cabin, QVector3D(0.75f, 0.85f, 0.95f));

    const float wheelR = 0.42f, wheelW = 0.32f;
    const float wx = 0.95f, wzFront = 1.15f, wzBack = -1.15f;
    QVector3D wheelPositions[4] = {
        {  wx, 0.42f, wzFront}, { -wx, 0.42f, wzFront},
        {  wx, 0.42f, wzBack }, { -wx, 0.42f, wzBack },
    };
    for (const QVector3D& wp : wheelPositions) {
        QMatrix4x4 wheel = carBase;
        wheel.translate(wp.x(), wp.y(), wp.z());
        wheel.rotate(90.0f, 0, 0, 1);
        wheel.translate(0.0f, -wheelW / 2.0f, 0.0f);
        wheel.scale(wheelR, wheelW, wheelR);
        drawObject(m_cylinder, wheel, QVector3D(0.08f, 0.08f, 0.08f));
    }

    m_program.release();
    painter.endNativePainting();

    // HUD overlay.
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(Qt::white);
    QFont font = painter.font();
    font.setPointSize(13);
    font.setBold(true);
    painter.setFont(font);
    float kmh = std::fabs(m_speed) * 3.6f;
    painter.drawText(16, 26, QString("מהירות: %1 קמ\"ש").arg(int(kmh)));
    painter.drawText(16, 48, QString("הקפות: %1").arg(m_lapCount));
    font.setPointSize(10);
    font.setBold(false);
    painter.setFont(font);
    painter.drawText(16, height() - 16, "חצים / WASD לנהיגה  |  Esc ליציאה");
}
