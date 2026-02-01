cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED on)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
endif()

project(opencv_project)

find_package(OpenCV REQUIRED)

add_executable(opencv_project src/main.cpp)
target_include_directories(opencv_project PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv_project PRIVATE ${OpenCV_LIBS})
