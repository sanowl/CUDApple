// Kernel for matrix addition
__global__ void matrixAdd(float *A, float *B, float *C, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

#ifndef __device__
#define __device__
#endif

#ifndef __shared__
#define __shared__
#endif

// Define CUDA synchronization functions
extern "C" __device__ void __syncthreads();

// Define CUDA built-in variables
struct uint3 {
    unsigned int x, y, z;
};

struct dim3 {
    unsigned int x, y, z;
};

#define TILE_SIZE 16

// Thread and block variables
static __device__ uint3 threadIdx;
static __device__ uint3 blockIdx;
static __device__ dim3 blockDim;

// CUDA vector type definition for Metal compatibility
struct float4 {
    float x, y, z, w;
};

__device__ float4 make_float4(float x, float y, float z, float w) {
    float4 v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;
    return v;
}

// Original matrix addition kernel
__global__ void matrixAdd(float *A, float *B, float *C, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

// Enhanced matrix addition with error checking
__global__ void matrixAddSafe(const float *A, const float *B, float *C, int M, int N)
{
    // Early return if invalid pointers
    if (A == nullptr || B == nullptr || C == nullptr) return;
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (row < M && col < N)
    {
        int idx = row * N + col;
        // Add bounds checking
        if (idx < M * N) {
            C[idx] = A[idx] + B[idx];
        }
    }
}

// Vectorized matrix addition for better performance
__global__ void matrixAddVectorized(const float4 *A, const float4 *B, float4 *C, int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread now processes 4 elements at once
    if (row < M && col < (N/4))  // N must be divisible by 4
    {
        int idx = row * (N/4) + col;
        float4 a = A[idx];
        float4 b = B[idx];
        
        // Vector operation
        C[idx] = make_float4(
            a.x + b.x,
            a.y + b.y,
            a.z + b.z,
            a.w + b.w
        );
    }
}

// Tiled matrix addition for better memory access patterns
__global__ void matrixAddTiled(const float *A, const float *B, float *C, int M, int N)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    if (row < M && col < N)
    {
        int idx = row * N + col;
        // Load data into shared memory
        tileA[threadIdx.y][threadIdx.x] = A[idx];
        tileB[threadIdx.y][threadIdx.x] = B[idx];
        
        __syncthreads();  // Ensure all threads have loaded their data
        
        // Perform addition using shared memory
        C[idx] = tileA[threadIdx.y][threadIdx.x] + tileB[threadIdx.y][threadIdx.x];
    }
}

// Main entry point that will be called by the Metal framework
extern "C" __global__ void mainKernel(const float *A, const float *B, float *C, int M, int N) {
    // This will automatically use the vectorized version for better performance
    matrixAddVectorized((const float4*)A, (const float4*)B, (float4*)C, M, N/4);
}