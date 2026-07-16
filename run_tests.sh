cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_FLAGS="-O2"
cmake --build build -j$(nproc)
build/run_tests
