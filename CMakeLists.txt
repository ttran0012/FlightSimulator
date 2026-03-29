cmake_minimum_required(VERSION 3.15)
project(FlightSimulator)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(FlightSimulator main.cpp)

target_include_directories(FlightSimulator PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(FlightSimulator PRIVATE ${OpenCV_LIBS})
