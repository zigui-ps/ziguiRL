cmake_minimum_required(VERSION 3.5.1)

get_filename_component(example_name MultiPendulum NAME)

project(${example_name})

set (CMAKE_CXX_FLAGS "-g")

set(required_components utils-urdf gui)
set(required_libraries dart dart-utils-urdf dart-gui)

if(DART_IN_SOURCE_BUILD)
  dart_build_tutorial_in_source(${example_name} LINK_LIBRARIES ${required_libraries})
  return()
endif()

find_package(DART 6.6.0 REQUIRED COMPONENTS ${required_components} CONFIG)

file(GLOB srcs "multipendulum.cpp")
#add_executable(${example_name} ${srcs})
add_library(${example_name} SHARED ${srcs})
target_link_libraries(${example_name} PUBLIC ${required_libraries})
