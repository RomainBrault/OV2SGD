cmake_minimum_required(VERSION 2.8)

project(ovdsgd)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

file(GLOB SOURCES "src/*.cpp")
file(GLOB INCLUDES "include/*.hpp")

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Eigen3 REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Boost 1.6.0 COMPONENTS python REQUIRED)
find_package(Numpy REQUIRED)

include_directories("${EIGEN3_INCLUDE_DIR}"
					"${PYTHON_INCLUDE_DIRS}"
	                "${Boost_INCLUDE_DIRS}"
	                "${NUMPY_INCLUDES}")
include_directories(include)
add_library(_ovdsgd SHARED ${SOURCES})
target_link_libraries(_ovdsgd ${PYTHON_LIBRARIES}
	                         ${Boost_LIBRARIES}
	                         ${NUMPY_LIBRARIES})
set_target_properties(_ovdsgd PROPERTIES PREFIX "")

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -fPIC")

install(TARGETS _ovdsgd DESTINATION lib)
install(FILES ${INCLUDES} DESTINATION include)

add_subdirectory(example)
