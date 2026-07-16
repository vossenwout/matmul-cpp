cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_FLAGS="-O2"
ln -sf build/compile_commands.json compile_commands.json
cmake --build build -j$(nproc)
build/run_tests
perf stat -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses ./build/matmul_cpp
