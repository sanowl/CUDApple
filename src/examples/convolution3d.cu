__global__ void conv3d(float *input, float *mask, float *result, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = size * size * size;
    
    if (idx < total_size)
    {
        // Convert linear index to 3D coordinates
        int z = idx / (size * size);
        int y = (idx / size) % size;
        int x = idx % size;
        
        // Simple 3x3x3 convolution
        float sum = 0.0f;
        
        // Only process interior points
        if (x > 0 && x < size-1 && y > 0 && y < size-1 && z > 0 && z < size-1)
        {
            // Apply 3x3x3 mask
            for (int k = -1; k <= 1; k++)
            {
                for (int j = -1; j <= 1; j++)
                {
                    for (int i = -1; i <= 1; i++)
                    {
                        int in_idx = (z+k) * size * size + (y+j) * size + (x+i);
                        int m_idx = (k+1) * 9 + (j+1) * 3 + (i+1);
                        sum += input[in_idx] * mask[m_idx];
                    }
                }
            }
            result[idx] = sum;
        }
        else
        {
            result[idx] = input[idx];
        }
    }
}

// Example usage function
void launch_conv3d(float *d_input, float *d_mask, float *d_result, 
                   int input_size, int mask_size)
{
    // Calculate grid and block dimensions
    dim3 blockSize(8, 8, 8);  // Adjust based on your GPU capabilities
    
    dim3 gridSize((input_size + blockSize.x - 1) / blockSize.x,
                  (input_size + blockSize.y - 1) / blockSize.y,
                  (input_size + blockSize.z - 1) / blockSize.z);
    
    conv3d<<<gridSize, blockSize>>>(d_input, d_mask, d_result, 
                                   input_size, mask_size);
    
    cudaDeviceSynchronize();
}