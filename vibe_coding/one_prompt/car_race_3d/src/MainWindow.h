#pragma once
#include <QMainWindow>
#include "GameView.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

private:
    GameView* m_gameView;
};
