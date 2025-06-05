use crate::parser::unified_ast::*;
use std::collections::HashMap;

pub struct MetalCodeGenerator {
    kernel_name: String,
    parameters: Vec<Parameter>,
    dimensions: usize,
    width: Option<usize>,
    height: Option<usize>,
}

impl MetalCodeGenerator {
    pub fn new(kernel: &KernelFunction) -> Self {
        let (dimensions, width, height) = Self::analyze_kernel_dimensions(kernel);
        
        Self {
            kernel_name: kernel.name.clone(),
            parameters: kernel.parameters.clone(),
            dimensions,
            width,
            height,
        }
    }
    
    fn analyze_kernel_dimensions(kernel: &KernelFunction) -> (usize, Option<usize>, Option<usize>) {
        // Analyze the AST to determine if this is 1D or 2D
        let has_2d_indexing = Self::check_for_2d_indexing(&kernel.body);
        
        if has_2d_indexing {
            (2, Some(32), Some(24)) // Default 2D dimensions
        } else {
            (1, None, None)
        }
    }
    
    fn check_for_2d_indexing(block: &Block) -> bool {
        for stmt in &block.statements {
            if Self::statement_has_2d_indexing(stmt) {
                return true;
            }
        }
        false
    }
    
    fn statement_has_2d_indexing(stmt: &Statement) -> bool {
        match stmt {
            Statement::Assign(Assignment { target, value }) => {
                Self::expression_has_2d_indexing(target) || Self::expression_has_2d_indexing(value)
            }
            Statement::IfStmt { condition, body } => {
                Self::expression_has_2d_indexing(condition) || Self::check_for_2d_indexing(body)
            }
            Statement::ForLoop { init, condition, increment, body } => {
                Self::statement_has_2d_indexing(init) || 
                Self::expression_has_2d_indexing(condition) ||
                Self::statement_has_2d_indexing(increment) ||
                Self::check_for_2d_indexing(body)
            }
            _ => false,
        }
    }
    
    fn expression_has_2d_indexing(expr: &Expression) -> bool {
        match expr {
            Expression::ThreadIdx(Dimension::Y) | 
            Expression::ThreadIdx(Dimension::Z) |
            Expression::BlockIdx(Dimension::Y) | 
            Expression::BlockIdx(Dimension::Z) => true,
            Expression::BinaryOp(left, _, right) => {
                Self::expression_has_2d_indexing(left) || Self::expression_has_2d_indexing(right)
            }
            Expression::ArrayAccess { array, index } => {
                Self::expression_has_2d_indexing(array) || Self::expression_has_2d_indexing(index)
            }
            _ => false,
        }
    }
    
    pub fn generate_metal_shader(&self, kernel: &KernelFunction) -> String {
        let mut metal_code = String::new();
        
        metal_code.push_str("#include <metal_stdlib>\n");
        metal_code.push_str("#include <metal_math>\n");
        metal_code.push_str("using namespace metal;\n\n");
        
        // Generate kernel signature
        metal_code.push_str(&format!("kernel void {}(", self.kernel_name));
        
        // Convert parameters to Metal buffer bindings
        for (i, param) in self.parameters.iter().enumerate() {
            if i > 0 {
                metal_code.push_str(",\n                    ");
            }
            metal_code.push_str(&self.convert_parameter(param, i));
        }
        
        // Add thread indexing parameter
        if self.dimensions == 1 {
            metal_code.push_str(",\n                    uint index [[thread_position_in_grid]])");
        } else {
            metal_code.push_str(",\n                    uint2 index [[thread_position_in_grid]])");
        }
        
        metal_code.push_str(" {\n");
        
        // Generate body by converting CUDA statements to Metal
        metal_code.push_str(&self.convert_kernel_body(&kernel.body));
        
        metal_code.push_str("}\n");
        metal_code
    }
    
    fn convert_parameter(&self, param: &Parameter, index: usize) -> String {
        match &param.param_type {
            Type::Pointer(inner_type) => {
                let metal_type = self.convert_type(inner_type);
                // Assume last buffer is output, others are input
                if index == self.parameters.len() - 1 || param.name == "result" || param.name == "output" || param.name == "c" {
                    format!("device {}* {} [[buffer({})]]", metal_type, param.name, index)
                } else {
                    format!("device const {}* {} [[buffer({})]]", metal_type, param.name, index)
                }
            }
            _ => {
                let metal_type = self.convert_type(&param.param_type);
                format!("constant {}& {} [[buffer({})]]", metal_type, param.name, index)
            }
        }
    }
    
    fn convert_type(&self, cuda_type: &Type) -> &str {
        match cuda_type {
            Type::Int => "int",
            Type::Float => "float",
            Type::Void => "void",
            Type::Vector(base, size) => {
                match (self.convert_type(base), size) {
                    ("float", 2) => "float2",
                    ("float", 3) => "float3", 
                    ("float", 4) => "float4",
                    ("int", 2) => "int2",
                    ("int", 3) => "int3",
                    ("int", 4) => "int4",
                    _ => "float",
                }
            }
            _ => "float",
        }
    }
    
    fn convert_kernel_body(&self, block: &Block) -> String {
        let mut body = String::new();
        
        // Add bounds checking at the start
        if self.dimensions == 1 {
            body.push_str("    uint i = index;\n");
            // Find array size parameter
            if let Some(size_param) = self.parameters.iter().find(|p| matches!(p.param_type, Type::Int)) {
                body.push_str(&format!("    if (i >= {}) return;\n", size_param.name));
            }
        } else {
            body.push_str("    uint x = index.x;\n");
            body.push_str("    uint y = index.y;\n");
            // Add 2D bounds checking if width/height parameters exist
            let width_param = self.parameters.iter().find(|p| p.name == "width" || p.name == "w");
            let height_param = self.parameters.iter().find(|p| p.name == "height" || p.name == "h");
            
            if let (Some(w), Some(h)) = (width_param, height_param) {
                body.push_str(&format!("    if (x >= {} || y >= {}) return;\n", w.name, h.name));
            }
        }
        
        body.push_str("\n");
        
        // Convert each statement
        for statement in &block.statements {
            body.push_str(&self.convert_statement(statement));
        }
        
        body
    }
    
    fn convert_statement(&self, stmt: &Statement) -> String {
        match stmt {
            Statement::Assign(assignment) => {
                format!("    {} = {};\n", 
                    self.convert_expression(&assignment.target),
                    self.convert_expression(&assignment.value))
            }
            Statement::IfStmt { condition, body } => {
                format!("    if ({}) {{\n{}\n    }}\n",
                    self.convert_expression(condition),
                    self.indent_body(&self.convert_kernel_body(body)))
            }
            Statement::ForLoop { init, condition, increment, body } => {
                format!("    for ({}; {}; {}) {{\n{}\n    }}\n",
                    self.convert_statement_inline(init),
                    self.convert_expression(condition),
                    self.convert_statement_inline(increment),
                    self.indent_body(&self.convert_kernel_body(body)))
            }
            Statement::CompoundAssign { target, operator, value } => {
                format!("    {} {}= {};\n",
                    self.convert_expression(target),
                    self.convert_operator(operator),
                    self.convert_expression(value))
            }
            Statement::AtomicOperation { operation, target, value } => {
                let metal_op = match operation {
                    AtomicOp::Add => "atomic_fetch_add_explicit",
                    AtomicOp::Sub => "atomic_fetch_sub_explicit",
                    AtomicOp::Exchange => "atomic_exchange_explicit",
                    _ => "atomic_fetch_add_explicit",
                };
                format!("    {}(&{}, {}, memory_order_relaxed);\n",
                    metal_op,
                    self.convert_expression(target),
                    self.convert_expression(value))
            }
            Statement::SyncThreads => {
                "    threadgroup_barrier(mem_flags::mem_threadgroup);\n".to_string()
            }
            Statement::VariableDecl(decl) => {
                let metal_type = self.convert_type(&decl.var_type);
                if let Some(init) = &decl.initializer {
                    format!("    {} {} = {};\n", 
                        metal_type, 
                        decl.name, 
                        self.convert_expression(init))
                } else {
                    format!("    {} {};\n", metal_type, decl.name)
                }
            }
            _ => "    // Unsupported statement\n".to_string(),
        }
    }
    
    fn convert_statement_inline(&self, stmt: &Statement) -> String {
        match stmt {
            Statement::VariableDecl(decl) => {
                let metal_type = self.convert_type(&decl.var_type);
                if let Some(init) = &decl.initializer {
                    format!("{} {} = {}", 
                        metal_type, 
                        decl.name, 
                        self.convert_expression(init))
                } else {
                    format!("{} {}", metal_type, decl.name)
                }
            }
            Statement::Assign(assignment) => {
                format!("{} = {}", 
                    self.convert_expression(&assignment.target),
                    self.convert_expression(&assignment.value))
            }
            _ => self.convert_statement(stmt).trim().to_string()
        }
    }
    
    fn indent_body(&self, body: &str) -> String {
        body.lines()
            .map(|line| format!("    {}", line))
            .collect::<Vec<_>>()
            .join("\n")
    }
    
    fn convert_expression(&self, expr: &Expression) -> String {
        match expr {
            Expression::Variable(name) => name.clone(),
            Expression::IntegerLiteral(val) => val.to_string(),
            Expression::FloatLiteral(val) => format!("{:.1}", val),
            Expression::BinaryOp(left, op, right) => {
                format!("({} {} {})",
                    self.convert_expression(left),
                    self.convert_operator(op),
                    self.convert_expression(right))
            }
            Expression::ThreadIdx(dim) => {
                match self.dimensions {
                    1 => "i".to_string(),
                    2 => match dim {
                        Dimension::X => "x".to_string(),
                        Dimension::Y => "y".to_string(),
                        _ => "0".to_string(),
                    },
                    _ => "index".to_string(),
                }
            }
            Expression::BlockIdx(dim) => {
                // For Metal, we simulate block indexing through thread position
                match self.dimensions {
                    1 => "0".to_string(), // Simplified - could be index / threadgroup_size
                    2 => match dim {
                        Dimension::X => "0".to_string(),
                        Dimension::Y => "0".to_string(),
                        _ => "0".to_string(),
                    },
                    _ => "0".to_string(),
                }
            }
            Expression::BlockDim(_) => "1".to_string(), // Simplified for Metal
            Expression::ArrayAccess { array, index } => {
                format!("{}[{}]",
                    self.convert_expression(array),
                    self.convert_expression(index))
            }
            Expression::MathFunction { name, arguments } => {
                let metal_name = self.convert_math_function(name);
                let args = arguments.iter()
                    .map(|arg| self.convert_expression(arg))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", metal_name, args)
            }
            Expression::Infinity => "INFINITY".to_string(),
            Expression::NegativeInfinity => "-INFINITY".to_string(),
        }
    }
    
    fn convert_operator(&self, op: &Operator) -> &str {
        match op {
            Operator::Add => "+",
            Operator::Subtract => "-",
            Operator::Multiply => "*",
            Operator::Divide => "/",
            Operator::LessThan => "<",
            Operator::LessThanEqual => "<=",
            Operator::GreaterThan => ">",
            Operator::GreaterThanEqual => ">=",
            Operator::Equal => "==",
            Operator::NotEqual => "!=",
            Operator::LogicalAnd => "&&",
            Operator::LogicalOr => "||",
        }
    }
    
    fn convert_math_function(&self, name: &str) -> &str {
        match name {
            "sin" => "sin",
            "cos" => "cos", 
            "tan" => "tan",
            "exp" => "exp",
            "log" => "log",
            "sqrt" => "sqrt",
            "pow" => "pow",
            "max" => "max",
            "min" => "min",
            "abs" => "abs",
            "floor" => "floor",
            "ceil" => "ceil",
            "round" => "round",
            _ => name,
        }
    }
    
    pub fn generate_swift_runner(&self, kernel: &KernelFunction) -> String {
        let metal_shader = self.generate_metal_shader(kernel);
        let metal_runner_class = self.generate_metal_runner_class();
        let parameter_init = self.generate_parameter_init();
        let kernel_call = self.generate_kernel_call();
        
        format!(r#"import Metal
import Foundation

{}

print("\n=== CUDApple Kernel Execution ===")
print("• Emulating CUDA kernel: {}")

{}

do {{
    let runner = try MetalKernelRunner(shaderCode: """
{}
""", kernelName: "{}")
    
    let startTime = CFAbsoluteTimeGetCurrent()
    
    {}
    
    let endTime = CFAbsoluteTimeGetCurrent()
    print("• Kernel execution completed in \(String(format: "%.3f", (endTime - startTime) * 1000))ms")
    
    print("\n=== Results ===")
    print("• First 5 output values:")
    for i in 0..<min(5, result.count) {{
        print("  [\(i)]: \(result[i])")
    }}
    
    // Verify correctness for known operations
    if result.count >= 5 {{
        print("\n• Sample verification:")
        print("  Expected: a[0] + b[0] = result[0]")
        print("  Got: result[0] = \(result[0])")
    }}
}} catch {{
    print("\n[ERROR] \(error)")
}}
"#, metal_runner_class, self.kernel_name, parameter_init, metal_shader, self.kernel_name, kernel_call)
    }
    
    fn generate_metal_runner_class(&self) -> String {
        r#"class MetalKernelRunner {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    
    init(shaderCode: String, kernelName: String) throws {
        print("\n=== Metal Device Detection ===")
        
        let devices = MTLCopyAllDevices()
        guard !devices.isEmpty else {
            throw MetalError.deviceNotFound
        }
        
        if let selectedDevice = devices.first(where: { $0.name.contains("Apple") }) {
            print("• Using device: \(selectedDevice.name)")
            self.device = selectedDevice
        } else {
            self.device = devices[0]
            print("• Using device: \(devices[0].name)")
        }
        
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue
        
        let compileOptions = MTLCompileOptions()
        compileOptions.fastMathEnabled = false
        compileOptions.languageVersion = .version2_4
        
        let library = try device.makeLibrary(source: shaderCode, options: compileOptions)
        guard let function = library.makeFunction(name: kernelName) else {
            throw MetalError.functionNotFound
        }
        
        self.pipeline = try device.makeComputePipelineState(function: function)
    }
    
    func executeKernel(inputs: [(data: Any, type: Any.Type)], outputType: Float.Type) throws -> [Float] {
        guard !inputs.isEmpty else { throw MetalError.invalidInput }
        
        var buffers: [MTLBuffer] = []
        
        let problemSize = if let firstArray = inputs[0].data as? [Float] {
            firstArray.count
        } else {
            throw MetalError.invalidInput
        }
        
        // Allocate buffers
        for (index, input) in inputs.enumerated() {
            if let array = input.data as? [Float] {
                guard let buffer = device.makeBuffer(bytes: array,
                                                   length: MemoryLayout<Float>.stride * array.count,
                                                   options: .storageModeShared) else {
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
            } else if let scalar = input.data as? Int {
                let uint32Value = UInt32(scalar)
                guard let buffer = device.makeBuffer(bytes: [uint32Value],
                                                   length: MemoryLayout<UInt32>.size,
                                                   options: .storageModeShared) else {
                    throw MetalError.bufferAllocationFailed
                }
                buffers.append(buffer)
            }
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        
        for (index, buffer) in buffers.enumerated() {
            computeEncoder.setBuffer(buffer, offset: 0, index: index)
        }
        
        let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let gridSize = MTLSize(width: (problemSize + 255) / 256, height: 1, depth: 1)
        
        computeEncoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        if let error = commandBuffer.error {
            throw MetalError.executionFailed
        }
        
        // Read results from output buffer (assuming last buffer is output)
        let outputBuffer = buffers[buffers.count - 2] // Typically the result buffer
        let outputPtr = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: outputPtr, count: problemSize))
    }
}

enum MetalError: Error {
    case deviceNotFound
    case commandQueueCreationFailed
    case functionNotFound
    case encoderCreationFailed
    case bufferAllocationFailed
    case invalidInput
    case executionFailed
}"#.to_string()
    }
    
    fn generate_parameter_init(&self) -> String {
        match self.dimensions {
            1 => {
                r#"// Initialize 1D test data
let n = 1024
let a = Array(0..<n).map { Float($0) }
let b = Array(0..<n).map { Float($0 * 2) }
let c = Array(repeating: Float(0), count: n)
let size = n

print("• Created input arrays with \(n) elements")"#.to_string()
            }
            2 => {
                format!(r#"// Initialize 2D matrix data
let width = {}
let height = {}
let totalElements = width * height
let matrixA = Array(0..<totalElements).map {{ Float($0) }}
let matrixB = Array(0..<totalElements).map {{ Float($0 * 2) }}
let matrixC = Array(repeating: Float(0), count: totalElements)

print("• Created \(width)x\(height) matrices (\(totalElements) elements)")"#,
                    self.width.unwrap_or(32),
                    self.height.unwrap_or(24))
            }
            _ => "// Unknown dimension".to_string(),
        }
    }
    
    fn generate_kernel_call(&self) -> String {
        match self.dimensions {
            1 => {
                r#"let inputs: [(data: Any, type: Any.Type)] = [
    (data: a, type: [Float].self),
    (data: b, type: [Float].self),
    (data: c, type: [Float].self),
    (data: size, type: Int.self)
]

let result = try runner.executeKernel(inputs: inputs, outputType: Float.self)"#.to_string()
            }
            2 => {
                r#"let inputs: [(data: Any, type: Any.Type)] = [
    (data: matrixA, type: [Float].self),
    (data: matrixB, type: [Float].self),
    (data: matrixC, type: [Float].self),
    (data: width, type: Int.self),
    (data: height, type: Int.self)
]

let result = try runner.executeKernel(inputs: inputs, outputType: Float.self)"#.to_string()
            }
            _ => "// Unknown kernel call".to_string(),
        }
    }
}