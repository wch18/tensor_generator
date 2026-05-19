cmake_minimum_required(VERSION 3.18)
project(tensor_generator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find libtorch
list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(Torch REQUIRED)

# Library: sources live under src/
add_library(tensor_generator_lib
    src/special_value.cpp
    src/tensor_generator.cpp
    src/strategies.cpp
)
target_include_directories(tensor_generator_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(tensor_generator_lib PUBLIC ${TORCH_LIBRARIES})

# GTest (built from source)
set(GTEST_ROOT "/opt/gtest")
find_package(GTest REQUIRED PATHS ${GTEST_ROOT} NO_DEFAULT_PATH)

# Test executable: source lives under test/
add_executable(tensor_generator_test
    test/tensor_generator_test.cpp
)
target_include_directories(tensor_generator_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)
target_link_libraries(tensor_generator_test
    tensor_generator_lib
    GTest::gtest
    GTest::gtest_main
    ${TORCH_LIBRARIES}
    pthread
)

# Test executable for convenience/strategies APIs
add_executable(test_convenience_strategies
    test/test_convenience_strategies.cpp
)
target_include_directories(test_convenience_strategies PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)
target_link_libraries(test_convenience_strategies
    tensor_generator_lib
    GTest::gtest
    GTest::gtest_main
    ${TORCH_LIBRARIES}
    pthread
)
