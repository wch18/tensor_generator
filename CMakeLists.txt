cmake_minimum_required(VERSION 3.18)
project(tensor_generator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find libtorch
list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(Torch REQUIRED)

# Header-only library: no add_library needed
set(TENSOR_GEN_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TENSOR_GEN_LIBS ${TORCH_LIBRARIES} pthread)

# GTest (built from source)
set(GTEST_ROOT "/opt/gtest")
find_package(GTest REQUIRED PATHS ${GTEST_ROOT} NO_DEFAULT_PATH)

# Test executable: source lives under test/
add_executable(tensor_generator_test
    test/tensor_generator_test.cpp
)
target_include_directories(tensor_generator_test PRIVATE
    ${TENSOR_GEN_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)
target_link_libraries(tensor_generator_test
    GTest::gtest
    GTest::gtest_main
    ${TENSOR_GEN_LIBS}
)

# Test executable for convenience/strategies APIs
add_executable(test_convenience_strategies
    test/test_convenience_strategies.cpp
)
target_include_directories(test_convenience_strategies PRIVATE
    ${TENSOR_GEN_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)
target_link_libraries(test_convenience_strategies
    GTest::gtest
    GTest::gtest_main
    ${TENSOR_GEN_LIBS}
)

# RNG trajectory test (standalone, not gtest)
add_executable(test_rng_trajectory
    test/test_rng_trajectory.cpp
)
target_include_directories(test_rng_trajectory PRIVATE
    ${TENSOR_GEN_INCLUDE}
)
target_link_libraries(test_rng_trajectory
    ${TENSOR_GEN_LIBS}
)
