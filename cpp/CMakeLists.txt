cmake_minimum_required(VERSION 3.16)
project(gridworld_fast LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization + AVX2. On Windows/MSVC use /arch:AVX2; on Clang/GCC use -mavx2 -O3.
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4 /permissive-)
else()
    add_compile_options(-O3 -mavx2 -mfma -Wall -Wextra -Wpedantic
                        -ffast-math -fno-omit-frame-pointer)
endif()

# pybind11 — fetched via pip-installed pybind11 or via add_subdirectory if vendored.
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(gridworld_fast
    env_fast.cpp
    bindings.cpp
)

target_include_directories(gridworld_fast PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Install next to the Python package so `import gridworld_fast` works.
install(TARGETS gridworld_fast DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../src)
