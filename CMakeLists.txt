cmake_minimum_required(VERSION 2.6)
project(gpu_convolve_test)
add_definitions(-std=c++11)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

add_executable(gpu_convolve_test gpu_convolve_test.cpp)

target_link_libraries(gpu_convolve_test ${OpenCV_LIBS})
