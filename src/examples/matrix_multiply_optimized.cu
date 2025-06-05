// Optimized Matrix Multiplication using shared memory
__global__ void matrixMulOptimized(float *A, float *B, float *C, 
                                 int M, int N, int K)  // C[M,N] = A[M,K] * B[K,N]
{
    __shared__ float sharedA[32][32];
    __shared__ float sharedB[32][32];
    
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    // Identify this thread's output element
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    float sum = 0.0f;
    
    // Loop over A and B tiles required to compute C element
    for (int t = 0; t < (K-1)/32 + 1; t++) {
        // Load tiles into shared memory
        if (row < M && (t*32 + tx) < K)
            sharedA[ty][tx] = A[row*K + t*32 + tx];
        else
            sharedA[ty][tx] = 0.0f;
            
        if ((t*32 + ty) < K && col < N)
            sharedB[ty][tx] = B[(t*32 + ty)*N + col];
        else
            sharedB[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial dot product for this tile
        for (int k = 0; k < 32; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row*N + col] = sum;
    }
}
