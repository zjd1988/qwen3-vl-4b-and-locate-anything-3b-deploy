add_executable(locate-anything-server main.cpp)
target_link_libraries(locate-anything-server PRIVATE locate_anything ggml)
target_include_directories(locate-anything-server PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party/stb ${CMAKE_SOURCE_DIR}/third_party/cpp-httplib)
find_package(Threads REQUIRED)
target_link_libraries(locate-anything-server PRIVATE Threads::Threads)
