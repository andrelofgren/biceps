cmake_minimum_required(VERSION 3.10)
project(biceps_demo VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})

find_package(Eigen3 5.0 REQUIRED NO_MODULE)
find_package(Boost 1.74 REQUIRED)
find_package(Matplot++ REQUIRED)
find_library(biceps biceps)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
)

add_executable(biceps_demo ${CMAKE_CURRENT_SOURCE_DIR}/biceps_demo.cpp)
target_link_libraries(
    biceps_demo
    ${Boost_LIBRARIES}
    Eigen3::Eigen
    Matplot++::matplot
    ${biceps}
)
install(TARGETS biceps_demo DESTINATION bin)
