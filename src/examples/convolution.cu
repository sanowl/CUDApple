// 2D Convolution Kernel
__global__ void convolution2D(float *input, float *kernel, float *output, 
                            int width, int height, int kernelSize)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tx < width && ty < height) {
        float sum = 0.0f;
        int radius = kernelSize / 2;
        
        // Convolution operation
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int imgX = tx + kx;
                int imgY = ty + ky;
                
                // Boundary check
                if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
                    int kernelIndex = (ky + radius) * kernelSize + (kx + radius);
                    int imageIndex = imgY * width + imgX;
                    sum += input[imageIndex] * kernel[kernelIndex];
                }
            }
        }
        
        output[ty * width + tx] = sum;
    }
}
