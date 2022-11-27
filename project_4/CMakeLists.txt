cmake_minimum_required(VERSION 3.12)

add_definitions(-DWITH_AVX2)

set(CMAKE_C_STANDARD 11)

project(Matrix)
 
add_compile_options(-mfma -fopenmp -mavx2 -O3 )



ADD_EXECUTABLE(Matrix Matrix.c test.c Matrix.h)

find_package(OpenMP)
if(OpenMP_C_FOUND)
    message("OpenMP found.")
    target_link_libraries(Matrix PUBLIC OpenMP::OpenMP_C)
endif()