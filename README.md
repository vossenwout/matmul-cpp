# Matrix Multiplication Optimization

I want to speed up inference and training in my [neural-cpp](https://github.com/vossenwout/neural-cpp.git) "neural network from scratch" project. Because tensor multiplications are expensive and I was using a naive `ijk` algorithm, I wanted to experiment with how much I could accelerate them. 

## Benchmark Setup

### Test Data

I benchmark the multiplication of two randomly initialized matrices (A * B = C):

- A = 1024 x 1024 (float32) = 4.19 MB
- B = 1024 x 1024 (float32) = 4.19 MB

**GFLOPs** =  `(2 * M * K * N) / 1e9` = `(2 * 1024^3) / 1e9` = **2.14**

### Hardware

I tested this on a Windows PC under WSL.

**CPU**
- Model name: 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz
- Socket(s): 1
- Core(s) per socket: 4
- Thread(s) per core:   2
- Instruction set: AVX2 (256-bit vector registers)
- FMA units per core: 2 (for AVX2)
- Caches:
    - L1d: 192 KiB (4 instances)
    - L1i: 128 KiB (4 instances)
    - L2: 5 MiB (4 instances)
    - L3: 12 MiB (1 instance)

**Estimated peak GFLOPs/s**:

GFLOPs/s per core = `clock_GHz * SIMD_Lanes * 2 (FMA = mul+add) * FMA_UNITS_CORE`
GFLOPs/s per core = `3.00 * (256 / 32) * 2 * 2` = 96 

We have four physical cores, so the peak across all cores is 96 * 4 = 384.

> We will compare the benchmark results against this peak.


## Benchmark results


### Overview


| Algorithm | Flags | GFLOP/s | % peak | IPC | L1 dcache misses | Why it got faster |
|---|---|---|---|---|---|---|
| ijk | -O2 | 1.11 | 1.15 | 0.82 | 3230369464 | / |
| ikj | -O2 | 5.36 | 5.59 | 4.59 | 203160054 | Better cache-line utilization |
| ikj | -O3 -march=native | 19.15 | 19.95 | 1.64 | 202966008  | Vectorization (`vfmadd...ps`) |
| blocked_ikj | -O3 -march=native | 22.71 | 23.65 | 2.25 | 249726588   | Try to keep blocks in cache |
| parallel_blocked_ikj | -O3 -march=native | 61.66 | 16.05 | 1.36 | 221436868    | Process blocks in parallel|



### Individual improvements

You can find all my implementations in `/src/matmul/matrix.cpp`

#### 1: ijk

This is the "naive" implementation. We traverse B down a column with a stride of N * sizeof(float) bytes (1024 * 4 = 4096 bytes). This means consecutive accesses land on different cache lines, causing poor cache-line utilization and frequent cache misses.

```cpp
#include <cstddef>

void matmul_ijk(
   const float* A,
   const float* B,
   float* C,
   std::size_t M,
   std::size_t K,
   std::size_t N)
{
   for (std::size_t i = 0; i < M; ++i)
   {
       for (std::size_t j = 0; j < N; ++j)
       {
           float acc = 0.0f;

           for (std::size_t k = 0; k < K; ++k)
           {
               acc += A[i * K + k] * B[k * N + j];
           }

           C[i * N + j] = acc;
       }
   }
}
```


#### 2: ikj

Now both B and C are accessed in the inner loop with a stride of 1 byte, allowing us to make better use of each cache line.

```cpp
#include <cstddef>

void matmul_ikj(
   const float* A,
   const float* B,
   float* C,
   std::size_t M,
   std::size_t K,
   std::size_t N)
{
   for (std::size_t i = 0; i < M; ++i)
   {
       for (std::size_t k = 0; k < K; ++k)
       {
           float a = A[i * K + k];

           for (std::size_t j = 0; j < N; ++j)
           {
               C[i * N + j] += a * B[k * N + j];
           }
       }
   }
}
```



#### 3: ikj + compiler optimization

x86-64 gcc 13.3

- [Godbolt ikj -O2](https://godbolt.org/z/Ya8bYhdha)
- [Godbolt ikj -O3 -march=native](https://godbolt.org/z/9xTnhdoG8)
- [Godbolt ijk -O3 -march=native](https://godbolt.org/z/eeMvW8MsY)


**Instructions**
 - `vfmadd...ss`: one scalar float
 - `vfmadd...ps` with `xmm`: four floats
 - `vfmadd...ps` with `ymm`: eight floats


 Snippet:
```
....
.L5:
        cmp     QWORD PTR [rsp], 6
        jbe     .L15
        vbroadcastss    ymm2, xmm1
        xor     r8d, r8d
.L8:
        vmovups ymm0, YMMWORD PTR [rdx+r8]
        vfmadd213ps     ymm0, ymm2, YMMWORD PTR [rax+r8]
        vmovups YMMWORD PTR [rax+r8], ymm0
        add     r8, 32
        cmp     r8, r9
        jne     .L8
...

```

Looking at the generated assembly, we can see that building with `-O3 -march=native` causes the `ikj` loop to use `vfmadd...ps` instructions with `xmm` registers for four floats and `ymm` registers for eight floats. This vectorizes the matrix multiplication and increases throughput. The same is not true for the `ijk` loop, which only uses scalar `vfmadd...ss` instructions.


#### 4: ikj + blocks/tiling

Further improvements can be made by calculating the matrix multiplication in blocks. This allows us to try to keep a block of `block_size * block_size` in cache, which should speed up memory access.

The benefits seem to be greatest around a block size of 128 * 128. Smaller blocks incur more loop overhead and repeated loads, while larger blocks  may no longer fit comfortably in a core’s private L2 cache. However I could not measure L2 and L3 behavior under wsl.



| block_size | GFLOP/s | Single-Core Peak (%) | IPC | L1D dcache Misses |
|---|---|---|---|---|
| 16 | 13.44 | 14.00 | 3.03 | 422682134 |
| 32 | 17.89 | 18.63 | 2.60 | 395824762 |
| 64 | 20.11 | 20.95 | 2.31 | 291005844 |
| 128 | 22.71 | 23.65 | 2.25 | 249726588 |
| 256 | 19.17 | 19.97 | 1.74 | 226938780 |
| 512 | 14.84 | 15.46 | 1.36 | 215741916 |



#### 5: ikj + tiling

Another benefit of tiling is that we can schedule each of the M / block_size rows on a separate thread, allowing us to parallelize the work.


| threads | block_size | GFLOP/s | Single-Core Peak (%) | IPC | L1D dcache misses |
|---|---|---|---|---|---|
| 2 | 128 | 41.27 | 42.99 | 1.90 | 246204559 |
| 2 | 256 | 25.80 | 26.87 | 1.33 | 218728207 |
| 2 | 512 | 24.34 | 25.36 | 1.04 | 192582853 |
| 4 | 128 | 55.76 | 58.08 | 1.52 | 246210580 |
| 4 | 256 | 61.66 | 64.23 | 1.36 | 221436868 |
| 4 | 512 | 22.95 | 23.91 | 0.90 | 204261026 |
| 8 | 128 | 55.37 | 57.67 | 1.02 | 248019967 |
| 8 | 256 | 49.19 | 51.24 | 0.91 | 183128259 |
| 8 | 512 | 24.22 | 25.23 | 0.74 | 213131238 |


The comparison against the single-core peak is not entirely fair, as we are now using multiple cores. However, performance seems to be best when the number of threads equals the number of physical cores and we use a block size of 256, so each row (1024 / 256 = 4) can be processed by a separate thread.


## Conclusion + Extensions

Although we managed to increase performance by approximately 55 times, there still seem to be many possible improvements, given that we are only using 16–23% of the CPU's estimated peak GFLOP/s. I expect next improvements to be found by reducing the repeated access and storage of new values to C in the loop. 
