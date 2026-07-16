g++ -Iinclude -O3 -march=native -ffast-math -std=c++23 -S -masm=intel src/matmul/matrix.cpp -o build/matrix.s
