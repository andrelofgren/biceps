#
# Copyright (C) 2025 André Löfgren
#
# This file is part of Biceps.
#
# Biceps is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Biceps is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Biceps. If not, see <https://www.gnu.org/licenses/>.
#
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS} 
    ${Python3_NumPy_INCLUDE_DIRS}
)
add_library(
    biceps
    ${CMAKE_CURRENT_SOURCE_DIR}/logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interval_mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/structured_mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fem_1d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fem_2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fem_function_1d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fem_function_2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/poisson_fem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pstokes_fem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/free_surface_fem.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/time_integrator.cpp
    ${PYTHON_WRAPPER}
)

include_directories(
    ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)
target_link_libraries(
    biceps
    Eigen3::Eigen
    ${Boost_LIBRARIES}
    ${Python_LIBRARIES}
    ${eigenpy_LIBRARIES}
)
install(TARGETS biceps DESTINATION lib)
