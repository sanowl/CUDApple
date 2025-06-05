import Metal
import Foundation

class MetalKernelRunner {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var buffers: [String: MTLBuffer] = [:]

    init() throws {
        print("\n=== Metal Device Detection ===")
        print("• Scanning for compatible Metal devices...")
        
        let devices = MTLCopyAllDevices()
        guard !devices.isEmpty else {
            throw MetalError.deviceNotFound
        }
        
        // Try to find Apple Silicon device
        if let selectedDevice = devices.first(where: { $0.name.contains("Apple") }) {
            print("• Using device: \(selectedDevice.name)")
            print("  ├─ Recommended max threads per threadgroup: \(selectedDevice.maxThreadsPerThreadgroup)")
            print("  └─ Supports unified memory: \(selectedDevice.hasUnifiedMemory ? "Yes" : "No")")
            self.device = selectedDevice
        } else {
            throw MetalError.deviceNotFound
        }
        
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue
        
        // Create pipeline state from generated shader
        let compileOptions = MTLCompileOptions()
        compileOptions.fastMathEnabled = false
        compileOptions.languageVersion = .version2_4

        let library = try device.makeLibrary(
            source: """
            #include <metal_stdlib>
            #include <metal_math>
            using namespace metal;
            
            // Vector Addition Kernel - Transpiled from CUDA
            kernel void vectorAdd(device const float* a [[buffer(0)]],
                                device const float* b [[buffer(1)]],
                                device float* c [[buffer(2)]],
                                constant uint& n [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
                if (index < n) {
                    c[index] = a[index] + b[index];
                }
            }
            
            // Matrix Addition Kernel - 2D Example
            kernel void matrixAdd(device const float* a [[buffer(0)]],
                                device const float* b [[buffer(1)]],
                                device float* c [[buffer(2)]],
                                constant uint& width [[buffer(3)]],
                                constant uint& height [[buffer(4)]],
                                uint2 index [[thread_position_in_grid]]) {
                uint x = index.x;
                uint y = index.y;
                if (x < width && y < height) {
                    uint idx = y * width + x;
                    c[idx] = a[idx] + b[idx];
                }
            }
            
            // Scalar Multiplication Kernel
            kernel void scalarMultiply(device const float* input [[buffer(0)]],
                                     device float* output [[buffer(1)]],
                                     constant float& scalar [[buffer(2)]],
                                     constant uint& n [[buffer(3)]],
                                     uint index [[thread_position_in_grid]]) {
                if (index < n) {
                    output[index] = input[index] * scalar;
                }
            }
            """, 
            options: compileOptions)

        if let error = library.makeFunction(name: "vectorAdd")?.label {
            print("Function creation error: \(error)")
        }
        
        self.pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "vectorAdd")!)
    }
    
    func allocateBuffer<T>(_ data: [T], index: Int) -> MTLBuffer? {
        print("\n• Allocating buffer \(index)")
        print("  ├─ Elements: \(data.count)")
        print("  └─ Size: \(MemoryLayout<T>.stride * data.count) bytes")
        
        guard let buffer = device.makeBuffer(bytes: data,
                                           length: MemoryLayout<T>.stride * data.count,
                                           options: .storageModeShared) else {
            print("[ERROR] Failed to allocate buffer \(index)")
            return nil
        }
        print("• Successfully allocated buffer \(index)")
        return buffer
    }
    
    func run(inputs: [MTLBuffer], problemSize: Int) throws {
        // Add debug prints for input buffers
        print("\n=== Buffer Contents Before Kernel Execution ===")
        for (index, buffer) in inputs.enumerated() {
            let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
            print("Buffer \(index) first 5 elements:")
            for i in 0..<5 {
                print("  [\(i)]: \(ptr[i])")
            }
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        
        // Set buffers
        for (index, buffer) in inputs.enumerated() {
            computeEncoder.setBuffer(buffer, offset: 0, index: index)
        }
        
        let config = KernelConfig.calculate(
            problemSize: problemSize,
            dimensions: 1,  // 1D vector operation
            width: nil,
            height: nil
        )
        
        print("\n=== Kernel Configuration ===")
        print("Grid Size: \(config.gridSize)")
        print("Thread Group Size: \(config.threadGroupSize)")
        print("Problem Size: \(problemSize)")
        print("Total Threads: \(config.gridSize.width * config.gridSize.height * config.threadGroupSize.width * config.threadGroupSize.height)")
        
        // Verify configuration
        if (config.gridSize.width * config.gridSize.height * 
            config.threadGroupSize.width * config.threadGroupSize.height) < problemSize {
            print("Warning: Grid size might be insufficient for problem size")
        }
        
        computeEncoder.dispatchThreadgroups(config.gridSize, 
                                     threadsPerThreadgroup: config.threadGroupSize)
        
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        if let error = commandBuffer.error {
            print("Command Buffer Error: \(error)")
            throw MetalError.executionFailed
        }
        
        // Add verification after kernel execution
        print("\n=== Buffer Contents After Kernel Execution ===")
        for (index, buffer) in inputs.enumerated() {
            let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
            print("Buffer \(index) first 5 elements:")
            for i in 0..<5 {
                print("  [\(i)]: \(ptr[i])")
            }
        }
    }

    func executeKernel<T>(inputs: [(data: Any, type: Any.Type)], outputType: T.Type) throws -> [T] {
        // Validate inputs
        guard !inputs.isEmpty else { throw MetalError.invalidInput }
        
        var buffers: [MTLBuffer] = []
        
        let problemSize = if let firstArray = inputs[0].data as? [T] {
            firstArray.count
        } else {
            throw MetalError.invalidInput
        }
        
        // Allocate and copy input data
        for (index, input) in inputs.enumerated() {
            if let array = input.data as? [T] {
                guard let buffer = allocateBuffer(array, index: index) else {
                    throw MetalError.bufferAllocationFailed
                }
                buffers.append(buffer)
            } else if let scalar = input.data as? UInt32 {
                guard let buffer = device.makeBuffer(bytes: [scalar],
                                                   length: MemoryLayout<UInt32>.size,
                                                   options: .storageModeShared) else {
                    throw MetalError.bufferAllocationFailed
                }
                buffers.append(buffer)
            }
        }
        
        try self.run(inputs: buffers, problemSize: problemSize)
        
        // Read back the result from the output buffer (res)
        guard buffers.count > 2 else {
            throw MetalError.invalidInput
        }
        
        let outputBuffer = buffers[2] // res is the third buffer
        let outputPtr = outputBuffer.contents().assumingMemoryBound(to: T.self)
        return Array(UnsafeBufferPointer(start: outputPtr, count: problemSize))
    }

    func readOutput<T>(buffer: MTLBuffer, type: T.Type) -> [T] {
        let count = buffer.length / MemoryLayout<T>.stride
        let pointer = buffer.contents().bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}

enum MetalError: Error {
    case deviceNotFound
    case commandQueueCreationFailed
    case functionNotFound
    case encoderCreationFailed
    case invalidBufferSize
    case bufferAllocationFailed
    case invalidInput
    case executionFailed
}

struct KernelConfig {
    let gridSize: MTLSize
    let threadGroupSize: MTLSize
    
    static func calculate(problemSize: Int, dimensions: Int, width: Int?, height: Int?) -> KernelConfig {
        if dimensions == 1 {
            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let gridWidth = (problemSize + 255) / 256
            let gridSize = MTLSize(width: gridWidth, height: 1, depth: 1)
            return KernelConfig(gridSize: gridSize, threadGroupSize: threadGroupSize)
        } else {
            // For 2D matrices, use M and N directly
            guard let w = width, let h = height else {
                fatalError("2D kernels require explicit width (M) and height (N)")
            }
            
            let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
            let gridSize = MTLSize(
                width: (w + threadGroupSize.width - 1) / threadGroupSize.width,
                height: (h + threadGroupSize.height - 1) / threadGroupSize.height,
                depth: 1
            )
            return KernelConfig(gridSize: gridSize, threadGroupSize: threadGroupSize)
        }
    }
}

// MARK: - Main Execution Script

import Foundation

print("\n=== CUDApple Kernel Execution ===")
print("• Emulating CUDA kernel: vectorAdd")

// Initialize test data
let n = 1024
let a = Array(0..<n).map { Float($0) }           // [0, 1, 2, 3, ...]
let b = Array(0..<n).map { Float($0 * 2) }       // [0, 2, 4, 6, ...]
let c = Array(repeating: Float(0), count: n)     // Output array
let size = UInt32(n)

print("• Created input arrays with \(n) elements")
print("• Array A: [\(a[0]), \(a[1]), \(a[2]), ..., \(a[n-1])]")
print("• Array B: [\(b[0]), \(b[1]), \(b[2]), ..., \(b[n-1])]")

do {
    // Initialize our Metal kernel runner
    let runner = try MetalKernelRunner()
    let startTime = CFAbsoluteTimeGetCurrent()
    
    // Execute vector addition: c = a + b
    let inputs: [(data: Any, type: Any.Type)] = [
        (data: a, type: [Float].self),      // Input array A
        (data: b, type: [Float].self),      // Input array B  
        (data: c, type: [Float].self),      // Output array C
        (data: size, type: UInt32.self)     // Array size
    ]
    
    let result = try runner.executeKernel(inputs: inputs, outputType: Float.self)
    
    let endTime = CFAbsoluteTimeGetCurrent()
    print("• Kernel execution completed in \(String(format: "%.3f", (endTime - startTime) * 1000))ms")
    
    print("\n=== Results ===")
    print("• First 5 output values:")
    for i in 0..<5 {
        print("  [\(i)]: \(a[i]) + \(b[i]) = \(result[i])")
    }
    
    print("• Last 5 output values:")
    for i in (n-5)..<n {
        print("  [\(i)]: \(a[i]) + \(b[i]) = \(result[i])")
    }
    
    // Verify correctness
    let isCorrect = zip(zip(a, b), result).allSatisfy { inputs, output in
        abs(inputs.0 + inputs.1 - output) < 0.0001
    }
    
    print("\n• Verification: \(isCorrect ? "✅ PASSED" : "❌ FAILED")")
    
} catch {
    print("\n[ERROR] \(error)")
}