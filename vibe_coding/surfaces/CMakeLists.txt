cmake_minimum_required(VERSION 3.16)
project(surfaces LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)

qt_add_executable(surfaces
    src/main.cpp
    src/MainWindow.cpp
    src/MainWindow.h
    src/SurfaceView.cpp
    src/SurfaceView.h
    src/Surface.cpp
    src/Surface.h
    src/Derivation.cpp
    src/Derivation.h
    src/DiagramScene.cpp
    src/DiagramScene.h
    src/DerivationView3D.cpp
    src/DerivationView3D.h
    shaders.qrc
)

target_link_libraries(surfaces PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
)
