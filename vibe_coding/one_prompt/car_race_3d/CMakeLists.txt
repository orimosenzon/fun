cmake_minimum_required(VERSION 3.16)
project(car_race_3d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)

qt_add_executable(car_race_3d
    src/main.cpp
    src/MainWindow.cpp
    src/MainWindow.h
    src/GameView.cpp
    src/GameView.h
    src/Mesh.cpp
    src/Mesh.h
)

target_link_libraries(car_race_3d PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
)
