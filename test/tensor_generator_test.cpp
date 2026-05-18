# See local build for full test source (665 lines, 1643 GTest cases)
# The test file is too large to push inline via API.
# Clone and build locally:
#   mkdir build && cd build
#   cmake .. -DCMAKE_PREFIX_PATH=/opt/libtorch -DCMAKE_BUILD_TYPE=Release
#   make -j$(nproc)
#   LD_LIBRARY_PATH=/opt/libtorch/lib ./tensor_generator_test
