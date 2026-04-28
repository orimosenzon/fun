#include <QApplication>
#include <QSurfaceFormat>

#include "MainWindow.h"

int main(int argc, char* argv[]) {
    QSurfaceFormat fmt;
    fmt.setVersion(3, 3);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    fmt.setSamples(4);
    fmt.setDepthBufferSize(24);
    QSurfaceFormat::setDefaultFormat(fmt);

    QApplication app(argc, argv);
    app.setApplicationName("Surfaces");
    app.setOrganizationName("vibe_coding");

    MainWindow w;
    w.show();
    return app.exec();
}
