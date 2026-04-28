#pragma once

#include <QMainWindow>
#include <QListWidget>
#include <QLabel>
#include <QCheckBox>
#include <QPushButton>
#include <QTextBrowser>
#include <memory>
#include <vector>

#include "Surface.h"
#include "SurfaceView.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void onSurfaceChanged(int row);

private:
    void buildUi();
    void updateInfo(const Surface& s);

    QVector<std::shared_ptr<Surface>> surfaces_;
    SurfaceView* view_ = nullptr;
    QListWidget* list_ = nullptr;
    QLabel* info_ = nullptr;
    QTextBrowser* derivationText_ = nullptr;
    class DerivationView3D* derivation3d_ = nullptr;
    QCheckBox* autoRotate_ = nullptr;
};
