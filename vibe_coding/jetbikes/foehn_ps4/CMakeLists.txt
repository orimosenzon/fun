cmake_minimum_required(VERSION 3.16)
project(foehn_ps4 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# Shared code: simulation (the ported flight model) + renderer. Platform-independent.
set(SIM_SRC
  src/sim/terrain.cpp
  src/sim/bike.cpp
  src/sim/fcs.cpp)
set(RENDER_SRC
  src/render/renderer.cpp
  src/render/env.cpp
  src/render/terrain_r.cpp
  src/render/water_r.cpp
  src/render/weather_r.cpp
  src/render/fx.cpp
  src/render/post.cpp
  src/render/props_r.cpp
  src/render/shadow.cpp
  src/render/bike_mesh.cpp)

option(PS4 "Build the PS4 homebrew target (needs the OpenOrbis toolchain)" OFF)

if(PS4)
  # Built with the OpenOrbis toolchain file (see toolchain/ps4.cmake and toolchain/NOTES.md).
  add_executable(foehn
    src/main.cpp ${SIM_SRC} ${RENDER_SRC}
    src/platform/platform_ps4.cpp)
  # OpenOrbis provides GLES2 via Piglet and input via scePad; link flags come from the toolchain.
else()
  # PC host: X11 + EGL + GLESv2 (the same GLES2 renderer that the PS4 build uses).
  add_executable(foehn
    src/main.cpp ${SIM_SRC} ${RENDER_SRC}
    src/platform/platform_pc.cpp)
  target_link_libraries(foehn EGL GLESv2 X11)
endif()

target_include_directories(foehn PRIVATE src)

# A headless physics check (no graphics): verifies the ported flight model flies.
add_executable(test_sim src/test_sim.cpp ${SIM_SRC})
target_include_directories(test_sim PRIVATE src)
