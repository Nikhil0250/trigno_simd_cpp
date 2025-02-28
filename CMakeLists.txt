# Specify the minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Define the project name
project(trigno_simd_cpp)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Compiler optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -O3 -march=native")

# Include header files
include_directories(include)

# Find all source files in src/ directory
file(GLOB SOURCES "src/*.cpp")

# Define the executable
add_executable(trigno_benchmark ${SOURCES})

# Specify the output directory for the executable
set_target_properties(trigno_benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
