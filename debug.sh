cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-O0 -g3 -fno-omit-frame-pointer"
cmake --build build -j$(nproc)
