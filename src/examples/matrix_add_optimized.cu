

// Advanced matrix addition with shared memory, vectorization and error handling
#define BLOCK_SIZE 16
#define VECTOR_SIZE 4

// Helper function to check array bounds
__device__ inline bool checkBounds(int row, int col, int M, int N) {
    return (row < M && col < N);
}

// Basic matrix addition with error checking
__global__ void matrixAddSafe(const float *A, const float *B, float *C, 
                            int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (checkBounds(row, col, M, N)) {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

// Optimized matrix addition using shared memory
__global__ void matrixAddShared(const float *A, const float *B, float *C, 
                               int M, int N) 
{
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // Load data into shared memory
    if (checkBounds(row, col, M, N)) {
        int idx = row * N + col;
        tileA[threadIdx.y][threadIdx.x] = A[idx];
        tileB[threadIdx.y][threadIdx.x] = B[idx];
    } else {
        tileA[threadIdx.y][threadIdx.x] = 0.0f;
        tileB[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    // Compute result
    if (checkBounds(row, col, M, N)) {
        int idx = row * N + col;
        C[idx] = tileA[threadIdx.y][threadIdx.x] + 
                 tileB[threadIdx.y][threadIdx.x];
    }
}

// Vectorized matrix addition for better memory bandwidth
__global__ void matrixAddVec4(const float4 *A4, const float4 *B4, float4 *C4, 
                             int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int N4 = N / 4; // N must be divisible by 4
    
    if (row < M && col < N4) {
        int idx = row * N4 + col;
        float4 a = A4[idx];
        float4 b = B4[idx];
        
        C4[idx] = make_float4(
            a.x + b.x,
            a.y + b.y,
            a.z + b.z,
            a.w + b.w
        );
    }
}

// Fused matrix add with scaling (C = alpha*A + beta*B)
__global__ void matrixAddScaled(const float *A, const float *B, float *C,
                               float alpha, float beta, int M, int N) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (checkBounds(row, col, M, N)) {
        int idx = row * N + col;
        C[idx] = alpha * A[idx] + beta * B[idx];
    }
}
