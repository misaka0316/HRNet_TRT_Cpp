cmake_minimum_required(VERSION 3.16)

project(MISAKA_Project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA的安装路径
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# 设置CUDA的架构，根据您的GPU类型选择合适的架构
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_86)

include_directories(
    /usr/local/TensorRT-8.6.1.6/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/C_HRNet
)

add_library(tensorrt_infer_lib SHARED
    src/tensorrt_infer.cpp
)
add_executable(misaka_executable  
    tests/test_main.cpp
)
add_executable(angle  
    src/angle.cpp
)

add_executable(sequence_detection  
    src/sequence_detection.cpp
)
add_executable(postprocessing  
    src/postprocessing.cpp
)

file(GLOB TENSORRT_LIBRARIES /usr/local/TensorRT-8.6.1.6/lib/*.so*)
file(GLOB MISAKACUDA_LIBRARIES /usr/local/cuda/lib64/*.so*)
target_link_libraries(misaka_executable PRIVATE 
    ${TENSORRT_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${MISAKACUDA_LIBRARIES}
    ${CUDART_LIBRARY}
    tensorrt_infer_lib
)