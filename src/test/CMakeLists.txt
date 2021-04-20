cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(simnet)


option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(Torch_DIR /home/zhongsy/Downloads/libtorch/share/cmake/Torch)
find_package(OpenCV REQUIRED)       # 查找OpenCV
find_package(CUDA REQUIRED)


set(CMAKE_CXX_STANDARD 14)


message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(/home/zhongsy/Downloads/TensorRT-7.0.0.11/include)
include_directories(/home/zhongsy/Downloads/TensorRT-7.0.0.11/samples)
include_directories(/usr/local/cuda/include)
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    /home/zhongsy/Downloads/TensorRT-7.0.0.11/samples/common
)


link_directories(/usr/local/cuda/lib64)
link_directories(/home/zhongsy/Downloads/TensorRT-7.0.0.11/lib)

add_executable(build build_enigen.cc)
add_executable(load load_enigen.cc)


target_link_libraries(build   ${CUDA_LIBS} nvinfer cudart nvparsers nvonnxparser protobuf nvinfer_plugin) 

target_link_libraries(load   ${CUDA_LIBS} nvinfer cudart nvparsers nvonnxparser protobuf nvinfer_plugin ${OpenCV_LIBS}) 

