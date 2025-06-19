__global__ void conv3d(float *input, float *mask, float *result, 
                       int input_size, int mask_size)
{
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid_z = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Calculate output dimensions (assuming valid convolution)
    int output_size = input_size - mask_size + 1;
    
    if (tid_x < output_size && tid_y < output_size && tid_z < output_size) {
        float sum = 0.0f;
        int half_mask = mask_size / 2;
        
        // Convolution loop over the 3D mask window
        for (int mz = 0; mz < mask_size; mz++) {
            for (int my = 0; my < mask_size; my++) {
                for (int mx = 0; mx < mask_size; mx++) {
                    // Calculate input coordinates
                    int input_x = tid_x + mx;
                    int input_y = tid_y + my;
                    int input_z = tid_z + mz;
                    
                    // Check bounds (for valid convolution, this should always be true)
                    if (input_x < input_size && input_y < input_size && input_z < input_size) {
                        // Calculate linear indices
                        int input_idx = input_z * input_size * input_size + 
                                       input_y * input_size + input_x;
                        int mask_idx = mz * mask_size * mask_size + 
                                      my * mask_size + mx;
                        
                        // Accumulate convolution result
                        sum += input[input_idx] * mask[mask_idx];
                    }
                }
            }
        }
        
        // Store result
        int output_idx = tid_z * output_size * output_size + 
                        tid_y * output_size + tid_x;
        result[output_idx] = sum;
    }
}

// Alternative version with zero-padding (same size output as input)
__global__ void conv3d_padded(float *input, float *mask, float *result, 
                             int size, int mask_size)
{
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid_z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (tid_x < size && tid_y < size && tid_z < size) {
        float sum = 0.0f;
        int half_mask = mask_size / 2;
        
        // Convolution loop over the 3D mask window
        for (int mz = 0; mz < mask_size; mz++) {
            for (int my = 0; my < mask_size; my++) {
                for (int mx = 0; mx < mask_size; mx++) {
                    // Calculate input coordinates with offset for centering
                    int input_x = tid_x + mx - half_mask;
                    int input_y = tid_y + my - half_mask;
                    int input_z = tid_z + mz - half_mask;
                    
                    // Zero-padding: check if coordinates are within bounds
                    if (input_x >= 0 && input_x < size && 
                        input_y >= 0 && input_y < size && 
                        input_z >= 0 && input_z < size) {
                        
                        // Calculate linear indices
                        int input_idx = input_z * size * size + 
                                       input_y * size + input_x;
                        int mask_idx = mz * mask_size * mask_size + 
                                      my * mask_size + mx;
                        
                        // Accumulate convolution result
                        sum += input[input_idx] * mask[mask_idx];
                    }
                    // Outside bounds contributes 0 (zero-padding)
                }
            }
        }
        
        // Store result
        int output_idx = tid_z * size * size + tid_y * size + tid_x;
        result[output_idx] = sum;
    }
}

// Example usage function
void launch_conv3d(float *d_input, float *d_mask, float *d_result, 
                   int input_size, int mask_size, bool use_padding = true)
{
    // Calculate grid and block dimensions
    dim3 blockSize(8, 8, 8);  // Adjust based on your GPU capabilities
    
    int output_size = use_padding ? input_size : input_size - mask_size + 1;
    
    dim3 gridSize((output_size + blockSize.x - 1) / blockSize.x,
                  (output_size + blockSize.y - 1) / blockSize.y,
                  (output_size + blockSize.z - 1) / blockSize.z);
    
    if (use_padding) {
        conv3d_padded<<<gridSize, blockSize>>>(d_input, d_mask, d_result, 
                                              input_size, mask_size);
    } else {
        conv3d<<<gridSize, blockSize>>>(d_input, d_mask, d_result, 
                                       input_size, mask_size);
    }
    
    cudaDeviceSynchronize();
}

// Optimized version using shared memory (for small masks)
__global__ void conv3d_shared(float *input, float *mask, float *result, 
                             int size, int mask_size)
{
    // Shared memory for input tile (including halo)
    extern __shared__ float shared_input[];
    
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid_z = blockIdx.z * blockDim.z + threadIdx.z;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    int half_mask = mask_size / 2;
    int shared_dim = blockDim.x + mask_size - 1;
    
    // Load data into shared memory with halo
    for (int sz = tz; sz < shared_dim; sz += blockDim.z) {
        for (int sy = ty; sy < shared_dim; sy += blockDim.y) {
            for (int sx = tx; sx < shared_dim; sx += blockDim.x) {
                int global_x = blockIdx.x * blockDim.x + sx - half_mask;
                int global_y = blockIdx.y * blockDim.y + sy - half_mask;
                int global_z = blockIdx.z * blockDim.z + sz - half_mask;
                
                int shared_idx = sz * shared_dim * shared_dim + sy * shared_dim + sx;
                
                if (global_x >= 0 && global_x < size && 
                    global_y >= 0 && global_y < size && 
                    global_z >= 0 && global_z < size) {
                    int global_idx = global_z * size * size + global_y * size + global_x;
                    shared_input[shared_idx] = input[global_idx];
                } else {
                    shared_input[shared_idx] = 0.0f;  // Zero padding
                }
            }
        }
    }
    
    __syncthreads();
    
    // Perform convolution using shared memory
    if (tid_x < size && tid_y < size && tid_z < size) {
        float sum = 0.0f;
        
        for (int mz = 0; mz < mask_size; mz++) {
            for (int my = 0; my < mask_size; my++) {
                for (int mx = 0; mx < mask_size; mx++) {
                    int shared_x = tx + mx;
                    int shared_y = ty + my;
                    int shared_z = tz + mz;
                    
                    int shared_idx = shared_z * shared_dim * shared_dim + 
                                    shared_y * shared_dim + shared_x;
                    int mask_idx = mz * mask_size * mask_size + 
                                  my * mask_size + mx;
                    
                    sum += shared_input[shared_idx] * mask[mask_idx];
                }
            }
        }
        
        int output_idx = tid_z * size * size + tid_y * size + tid_x;
        result[output_idx] = sum;
    }
}