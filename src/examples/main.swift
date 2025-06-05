import Foundation
import Metal

// Function to select and run the appropriate kernel
func runKernel(kernelName: String, A: [Float], B: [Float], M: Int, N: Int) -> [Float] {
    let runner = try! MetalKernelRunner()
    
    // Prepare input data
    var C = Array(repeating: Float(0), count: M * N)
    
    switch kernelName {
    case "matrixAddVectorized":
        // Convert to float4 arrays for vectorized version
        let float4Count = (M * N + 3) / 4
        var A4 = Array(repeating: float4(x: 0, y: 0, z: 0, w: 0), count: float4Count)
        var B4 = Array(repeating: float4(x: 0, y: 0, z: 0, w: 0), count: float4Count)
        var C4 = Array(repeating: float4(x: 0, y: 0, z: 0, w: 0), count: float4Count)
        
        // Pack float arrays into float4 arrays
        for i in 0..<(M * N / 4) {
            A4[i] = float4(x: A[i*4], y: A[i*4+1], z: A[i*4+2], w: A[i*4+3])
            B4[i] = float4(x: B[i*4], y: B[i*4+1], z: B[i*4+2], w: B[i*4+3])
        }
        
        // Run vectorized kernel
        try! runner.executeKernel(inputs: [(A4, Float4.self), (B4, Float4.self)], outputType: Float4.self)
        
        // Unpack results
        for i in 0..<(M * N / 4) {
            C[i*4] = C4[i].x
            C[i*4+1] = C4[i].y
            C[i*4+2] = C4[i].z
            C[i*4+3] = C4[i].w
        }
        
    default:
        // Run regular kernel
        try! runner.executeKernel(inputs: [(A, Float.self), (B, Float.self)], outputType: Float.self)
    }
    
    return C
}

// Test the kernels
let M = 16
let N = 16
let A = Array(0..<Float(M * N))
let B = Array(0..<Float(M * N)).map { $0 * 2 }

print("\nRunning matrix addition...")
let result = runKernel(kernelName: "matrixAddVectorized", A: A, B: B, M: M, N: N)

// Print first few results
print("\nFirst few results:")
for i in 0..<min(5, result.count) {
    print("C[\(i)] = \(result[i])")
}
