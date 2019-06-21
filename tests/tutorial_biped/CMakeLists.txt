cmake_minimum_required(VERSION 3.5.1)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++17)
add_compile_options(-Wdeprecated-declarations)
add_compile_options(-lm)
add_compile_options(-g)

get_filename_component(example_name ${CMAKE_CURRENT_LIST_DIR} NAME)

project(${example_name})

set(required_components collision-bullet utils-urdf gui)
set(required_libraries dart dart-collision-bullet dart-utils-urdf dart-gui)

if(DART_IN_SOURCE_BUILD)
  dart_build_tutorial_in_source(${example_name} LINK_LIBRARIES ${required_libraries})
  return()
endif()

find_package(DART 6.6.0 REQUIRED COMPONENTS ${required_components} CONFIG)

set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake_module")
find_package(TinyXML REQUIRED)

include_directories(${TinyXML_INCLUDE_DIRS})

file(GLOB srcs "*.cpp" "*.hpp")
add_executable(${example_name} ${srcs})
target_link_libraries(${example_name} PUBLIC ${required_libraries} ${TinyXML_LIBRARIES})
