export OMP_NUM_THREADS=2 # for the parallelized tiled approach
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_FLAGS="-O3 -march=native -ffast-math"
ln -sf build/compile_commands.json compile_commands.json
cmake --build build -j$(nproc)
build/run_tests
perf stat -e task-clock,cycles,instructions,L1-dcache-load-misses,LLC-load-misses ./build/matmul_cpp


