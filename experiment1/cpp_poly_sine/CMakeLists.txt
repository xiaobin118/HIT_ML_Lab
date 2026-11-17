cmake_minimum_required(VERSION 3.16)
project(poly_sine_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(poly_sine
    src/main.cpp
    src/data.cpp
    src/matrix.cpp
    src/polynomial_features.cpp
    src/linear_algebra.cpp
)

target_include_directories(poly_sine PRIVATE include)
