#include "MainWindow.h"

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QString::fromUtf8("מרוץ מכוניות תלת־ממדי"));
    m_gameView = new GameView(this);
    setCentralWidget(m_gameView);
    resize(1280, 800);
}
