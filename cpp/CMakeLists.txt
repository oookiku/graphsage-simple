cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

# libtorch
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/download/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# boost
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED thread)
if(Boost_FOUND)
    message(STATUS "!!! BOOST FOUND !!!")
else()
    message(STATUS "!!! BOOST NOT FOUND !!!")
endif()
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

add_executable(example-app main.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
target_link_libraries(example-app "${Boost_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)
