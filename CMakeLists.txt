cmake_minimum_required(VERSION 2.8)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0042 NEW)
endif()

option(USE_FLTO "Use link time optimisation" ON)
option(USE_OPENMP "Use OpenMP" ON)
option(US_TCMALLOC "Use tcmalloc" ON)
set(PYBIND11_PYTHON_VERSION ""
	CACHE STRING "Python version to use for compiling the example application")

project(ov2sgd)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

file(GLOB SOURCES "src/cpp/*.cpp")
file(GLOB PYSOURCES "src/pycpp/py_*.cpp")
file(GLOB INCLUDES "include/*.hpp")

include(CheckCXXCompilerFlag)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

set(PYBIND11_PYTHON_VERSION "3.5")
set(Python_ADDITIONAL_VERSIONS "3.5")

# Try to autodetect Python (can be overridden manually if needed)
find_package(PythonLibs ${PYBIND11_PYTHON_VERSION} REQUIRED)
find_package(PythonInterp ${PYBIND11_PYTHON_VERSION} REQUIRED)
string(REPLACE "+" "" PYTHONLIBS_VERSION_STRING "+${PYTHONLIBS_VERSION_STRING}")

include_directories("${PROJECT_SOURCE_DIR}/pybind11/include")
include_directories("${PYTHON_INCLUDE_DIRS}")
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/Eigen")

add_library(ov2sgd STATIC ${SOURCES})
add_library(_pyov2sgd SHARED ${PYSOURCES})

target_link_libraries(_pyov2sgd ov2sgd)
set_target_properties(_pyov2sgd PROPERTIES PREFIX "")

if (USE_OPENMP)
	find_package(OpenMP)
	if(OPENMP_FOUND)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	endif()
endif()

if (US_TCMALLOC)
	find_package(TcMalloc)
	if(Tcmalloc_FOUND)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
-ltcmalloc \
-fno-builtin-malloc -fno-builtin-calloc \
-fno-builtin-realloc -fno-builtin-free")
	endif()
endif()

message(STATUS CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -mtune=native \
-march=native -funroll-loops -ffast-math")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++")
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	CHECK_CXX_COMPILER_FLAG(-std=c++14 HAS_CPP14_FLAG)
	CHECK_CXX_COMPILER_FLAG(-std=c++11 HAS_CPP11_FLAG)

	if (HAS_CPP14_FLAG)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
	elseif (HAS_CPP11_FLAG)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	else()
		message(FATAL_ERROR "Unsupported compiler -- at least C++11 support is needed!")
	endif()

	# Enable link time optimization and set the default symbol
	# visibility to hidden (very important to obtain small binaries)
	if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
		# Default symbol visibility
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

		# Check for Link Time Optimization support
		if (USE_LTO)
			CHECK_CXX_COMPILER_FLAG(-flto HAS_LTO_FLAG)
			if (HAS_LTO_FLAG)
				set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
			endif()
		endif()
	endif()
endif()

if (WIN32)
	if (MSVC)
		# /bigobj is needed for bigger binding projects due to the limit to 64k
		# addressable sections. /MP enables multithreaded builds (relevant when
		# there are many files).
		set_target_properties(_pyov2sgd PROPERTIES COMPILE_FLAGS "/MP /bigobj ")

		if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
		  # Enforce size-based optimization and link time code generation on MSVC
		  # (~30% smaller binaries in experiments).
		  set_target_properties(_pyov2sgd APPEND_STRING PROPERTY COMPILE_FLAGS "/Os /GL ")
		  set_target_properties(_pyov2sgd APPEND_STRING PROPERTY LINK_FLAGS "/LTCG ")
		endif()
	endif()

	# .PYD file extension on Windows
	set_target_properties(_pyov2sgd PROPERTIES SUFFIX ".pyd")

	# Link against the Python shared library
	target_link_libraries(_pyov2sgd ${PYTHON_LIBRARY})
elseif (UNIX)
	# It's quite common to have multiple copies of the same Python version
	# installed on one's system. E.g.: one copy from the OS and another copy
	# that's statically linked into an application like Blender or Maya.
	# If we link our plugin library against the OS Python here and import it
	# into Blender or Maya later on, this will cause segfaults when multiple
	# conflicting Python instances are active at the same time (even when they
	# are of the same version).

	# Windows is not affected by this issue since it handles DLL imports
	# differently. The solution for Linux and Mac OS is simple: we just don't
	# link against the Python library. The resulting shared library will have
	# missing symbols, but that's perfectly fine -- they will be resolved at
	# import time.

	# .SO file extension on Linux/Mac OS
	set_target_properties(_pyov2sgd PROPERTIES SUFFIX ".so")

	# Strip unnecessary sections of the binary on Linux/Mac OS
	if(APPLE)
		set_target_properties(_pyov2sgd PROPERTIES MACOSX_RPATH ".")
		set_target_properties(_pyov2sgd PROPERTIES LINK_FLAGS "-undefined dynamic_lookup ")
		if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
		  	add_custom_command(TARGET _pyov2sgd POST_BUILD COMMAND strip -u -r ${PROJECT_BINARY_DIR}/_pyov2sgd.so)
		endif()
		else()
		if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
		 	add_custom_command(TARGET _pyov2sgd POST_BUILD COMMAND strip ${PROJECT_BINARY_DIR}/_pyov2sgd.so)
		endif()
	endif()
endif()


install(TARGETS ov2sgd DESTINATION "${PROJECT_SOURCE_DIR}/lib")
install(TARGETS _pyov2sgd DESTINATION "${PROJECT_SOURCE_DIR}/pyov2sgd")
install(FILES ${INCLUDES} DESTINATION include)

add_subdirectory(example)
