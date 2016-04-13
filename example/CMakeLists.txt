cmake_minimum_required(VERSION 2.8)

add_executable(synthetic_data main.cpp)
target_link_libraries(synthetic_data LINK_PUBLIC _ovdsgd)
install(TARGETS synthetic_data DESTINATION bin)