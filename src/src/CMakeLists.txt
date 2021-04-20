cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(simnet)


option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(Torch_DIR /home/zhongsy/Downloads/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)        # 查找libtorch
find_package(OpenCV REQUIRED)       # 查找OpenCV
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)




set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

#  设置调试环境
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")


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

add_executable(simnet pytorch.cc)
add_executable(build build_enigen.cc)
add_executable(load load_enigen.cc)
add_executable(load_1 load_enigen_1.cc)

# add_executable(build_cafee build_cafee.cc)


target_link_libraries(build  ${TORCH_LIBRARIES} ${CUDA_LIBS} nvinfer cudart nvparsers nvonnxparser protobuf nvinfer_plugin) 
# target_link_libraries(build_cafee  ${TORCH_LIBRARIES} ${CUDA_LIBS} nvinfer cudart nvparsers nvonnxparser protobuf nvinfer_plugin) 

target_link_libraries(load  ${TORCH_LIBRARIES} ${CUDA_LIBS} nvinfer cudart nvparsers nvonnxparser protobuf nvinfer_plugin ${OpenCV_LIBS}) 
target_link_libraries(load_1  ${TORCH_LIBRARIES} ${CUDA_LIBS} nvinfer cudart nvparsers nvonnxparser protobuf nvinfer_plugin ${OpenCV_LIBS}) 

target_link_libraries(simnet ${TORCH_LIBRARIES} ${OpenCV_LIBS}) 
