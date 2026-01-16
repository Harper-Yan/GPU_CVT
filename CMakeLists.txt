cmake_minimum_required(VERSION 3.18)
project(gpu_cvt LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(gpu_cvt main.cu)
set_target_properties(gpu_cvt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
