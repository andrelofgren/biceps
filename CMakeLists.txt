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
cmake_minimum_required(VERSION 3.10)
project(biceps VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(ENABLE_PYTHON ON CACHE BOOL "Enable Python support")
set(BUILD_SHARED_LIBS ON CACHE BOOL "Enable shared libs")
# set(ENABLE_DOCS ON CACHE BOOL "Build documentation")

# Check for Eigen3 and Boost (mandatory)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Boost 1.74 REQUIRED)

# Check for Python interface dependencies (optional)
find_package(Python COMPONENTS Development NumPy QUIET)

if(
    BUILD_SHARED_LIBS AND
    ENABLE_PYTHON AND
    Python_FOUND AND
    Python_Development_FOUND AND
    Python_NumPy_FOUND AND
)
    set(PYTHON_WRAPPER ${CMAKE_SOURCE_DIR}/src/pywrapper.cpp)
    message(STATUS "Building Python interface.")
else()
    message(STATUS "Python interface will not be built.")
endif()

# Check for unit testing dependency (optional)
find_package(Boost 1.74 COMPONENTS unit_test_framework QUIET)
if(ENABLE_TESTS AND Boost_UNIT_TEST_FRAMEWORK_FOUND)
    message(STATUS "Boost unit test framework found; enabling test target.")
    add_subdirectory(unit_tests)
elseif(ENABLE_TESTS AND NOT Boost_UNIT_TEST_FRAMEWORK_FOUND)
    message(WARNING "Boost unit test framework not found, but ENABLE_TESTS is ON; disabling test target.")
else()
    message(STATUS "test target disabled.")
endif()

# Project source files
add_subdirectory(src)

install(
    FILES 
        include/eigen_spmat_addons.hpp
        include/enums.hpp
        include/logger.hpp
        include/interval_mesh.hpp
        include/structured_mesh.hpp
        include/fem_1d.hpp
        include/fem_2d.hpp
        include/fem_function_1d.hpp
        include/fem_function_2d.hpp
        include/pstokes_fem.hpp
        include/poisson_fem.hpp
        include/free_surface_fem.hpp
        include/time_integrator.hpp
    DESTINATION
        include
)
