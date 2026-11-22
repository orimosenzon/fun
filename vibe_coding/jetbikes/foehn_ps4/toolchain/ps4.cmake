# CMake toolchain file for the PS4 homebrew build (OpenOrbis).
# Usage:  cmake -DPS4=ON -DCMAKE_TOOLCHAIN_FILE=toolchain/ps4.cmake -B build-ps4
#
# Requires the OpenOrbis toolchain installed and $OO_PS4_TOOLCHAIN set:
#   https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain
#
# This is a starting point. The exact flags/libraries track the OpenOrbis version installed;
# see the toolchain's own sample CMake/Makefiles and toolchain/NOTES.md.

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

if(NOT DEFINED ENV{OO_PS4_TOOLCHAIN})
  message(FATAL_ERROR "OO_PS4_TOOLCHAIN is not set. Install the OpenOrbis toolchain first (see toolchain/NOTES.md).")
endif()
set(OO $ENV{OO_PS4_TOOLCHAIN})

set(CMAKE_C_COMPILER   "${OO}/bin/linux/clang")
set(CMAKE_CXX_COMPILER "${OO}/bin/linux/clang++")

set(PS4_TARGET "--target=x86_64-pc-freebsd12-elf -fPIC -funwind-tables")
set(PS4_INCLUDES "-isysroot ${OO} -isystem ${OO}/include -isystem ${OO}/include/c++/v1")

set(CMAKE_CXX_FLAGS "${PS4_TARGET} ${PS4_INCLUDES} -DPS4_BUILD -std=c++17 -O2" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS   "${PS4_TARGET} ${PS4_INCLUDES} -DPS4_BUILD -O2" CACHE STRING "" FORCE)

# Linking to eboot.bin and packing to .pkg is done by the OpenOrbis create-fself / create-pkg
# tools as a post-build step; wire that in once the toolchain is installed.
