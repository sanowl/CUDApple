use crate::parser::unified_ast::*;

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
            Statement::IfStmt {
                condition, body, ..
            } => Self::expression_has_2d_indexing(condition) || Self::check_for_2d_indexing(body),
            Statement::ForLoop {
                init,
                condition,
                increment,
                body,
            } => {
                Self::statement_has_2d_indexing(init)
                    || Self::expression_has_2d_indexing(condition)
                    || Self::statement_has_2d_indexing(increment)
                    || Self::check_for_2d_indexing(body)
            }
            _ => false,
        }
    }

    fn expression_has_2d_indexing(expr: &Expression) -> bool {
        match expr {
            Expression::ThreadIdx(Dimension::Y)
            | Expression::ThreadIdx(Dimension::Z)
            | Expression::BlockIdx(Dimension::Y)
            | Expression::BlockIdx(Dimension::Z) => true,
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

        metal_code.push_str(&format!("kernel void {}(", self.kernel_name));

        for (i, param) in self.parameters.iter().enumerate() {
            if i > 0 {
                metal_code.push_str(",\n                    ");
            }
            metal_code.push_str(&self.convert_parameter(param, i));
        }

        metal_code.push_str(
            ",\n                    uint3 cuda_grid_position [[thread_position_in_grid]]",
        );
        metal_code.push_str(
            ",\n                    uint3 cuda_thread_idx [[thread_position_in_threadgroup]]",
        );
        metal_code.push_str(
            ",\n                    uint3 cuda_block_idx [[threadgroup_position_in_grid]]",
        );
        metal_code
            .push_str(",\n                    uint3 cuda_block_dim [[threads_per_threadgroup]]");
        metal_code
            .push_str(",\n                    uint3 cuda_threads_per_grid [[threads_per_grid]])");

        metal_code.push_str(" {\n");
        metal_code.push_str("    (void)cuda_grid_position;\n");
        metal_code.push_str("    (void)cuda_threads_per_grid;\n");

        metal_code.push_str(&self.convert_kernel_body(&kernel.body));

        metal_code.push_str("}\n");
        metal_code
    }

    fn convert_parameter(&self, param: &Parameter, index: usize) -> String {
        match &param.param_type {
            Type::Pointer(inner_type) => {
                let metal_type = self.convert_type(inner_type);
                if self.is_output_parameter(param, index) {
                    format!(
                        "device {}* {} [[buffer({})]]",
                        metal_type, param.name, index
                    )
                } else {
                    format!(
                        "device const {}* {} [[buffer({})]]",
                        metal_type, param.name, index
                    )
                }
            }
            _ => {
                let metal_type = self.convert_type(&param.param_type);
                format!(
                    "constant {}& {} [[buffer({})]]",
                    metal_type, param.name, index
                )
            }
        }
    }

    fn is_output_parameter(&self, param: &Parameter, index: usize) -> bool {
        if !matches!(param.param_type, Type::Pointer(_)) {
            return false;
        }

        let lower_name = param.name.to_ascii_lowercase();
        let name_looks_writable = matches!(
            lower_name.as_str(),
            "c" | "res" | "result" | "out" | "output" | "dst" | "destination"
        ) || lower_name.contains("output")
            || lower_name.contains("result");

        name_looks_writable || Some(index) == self.output_buffer_index()
    }

    fn output_buffer_index(&self) -> Option<usize> {
        self.parameters
            .iter()
            .enumerate()
            .rev()
            .find(|(_, param)| matches!(param.param_type, Type::Pointer(_)))
            .map(|(index, _)| index)
    }

    fn convert_type(&self, cuda_type: &Type) -> &str {
        match cuda_type {
            Type::Int => "int",
            Type::Float => "float",
            Type::Void => "void",
            Type::Vector(base, size) => match (self.convert_type(base), size) {
                ("float", 2) => "float2",
                ("float", 3) => "float3",
                ("float", 4) => "float4",
                ("int", 2) => "int2",
                ("int", 3) => "int3",
                ("int", 4) => "int4",
                _ => "float",
            },
            _ => "float",
        }
    }

    fn convert_kernel_body(&self, block: &Block) -> String {
        let mut body = String::new();

        for statement in &block.statements {
            body.push_str(&self.convert_statement(statement));
        }

        body
    }

    fn convert_statement(&self, stmt: &Statement) -> String {
        match stmt {
            Statement::Assign(assignment) => {
                format!(
                    "    {} = {};\n",
                    self.convert_expression(&assignment.target),
                    self.convert_expression(&assignment.value)
                )
            }
            Statement::IfStmt {
                condition,
                body,
                else_body,
            } => {
                let mut statement = format!(
                    "    if ({}) {{\n{}\n    }}",
                    self.convert_expression(condition),
                    self.indent_body(&self.convert_kernel_body(body))
                );

                if let Some(else_block) = else_body {
                    statement.push_str(&format!(
                        " else {{\n{}\n    }}",
                        self.indent_body(&self.convert_kernel_body(else_block))
                    ));
                }

                statement.push('\n');
                statement
            }
            Statement::ForLoop {
                init,
                condition,
                increment,
                body,
            } => {
                format!(
                    "    for ({}; {}; {}) {{\n{}\n    }}\n",
                    self.convert_statement_inline(init),
                    self.convert_expression(condition),
                    self.convert_statement_inline(increment),
                    self.indent_body(&self.convert_kernel_body(body))
                )
            }
            Statement::CompoundAssign {
                target,
                operator,
                value,
            } => {
                format!(
                    "    {} {}= {};\n",
                    self.convert_expression(target),
                    self.convert_operator(operator),
                    self.convert_expression(value)
                )
            }
            Statement::AtomicOperation {
                operation,
                target,
                value,
                ..
            } => {
                format!(
                    "    atomic_{}({}, {});\n",
                    match operation {
                        AtomicOp::Add => "fetch_add_explicit",
                        AtomicOp::Sub => "fetch_sub_explicit",
                        AtomicOp::Exchange => "exchange_explicit",
                        AtomicOp::Min => "fetch_min_explicit",
                        AtomicOp::Max => "fetch_max_explicit",
                        AtomicOp::And => "fetch_and_explicit",
                        AtomicOp::Or => "fetch_or_explicit",
                        AtomicOp::Xor => "fetch_xor_explicit",
                        AtomicOp::CAS => "compare_exchange_weak_explicit",
                        AtomicOp::Inc => "fetch_add_explicit",
                        AtomicOp::Dec => "fetch_sub_explicit",
                    },
                    self.convert_expression(target),
                    self.convert_expression(value)
                )
            }
            Statement::SyncThreads => {
                "    threadgroup_barrier(mem_flags::mem_threadgroup);\n".to_string()
            }
            Statement::VariableDecl(decl) => {
                let metal_type = self.convert_type(&decl.var_type);
                if let Some(init) = &decl.initializer {
                    format!(
                        "    {} {} = {};\n",
                        metal_type,
                        decl.name,
                        self.convert_expression(init)
                    )
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
                    format!(
                        "{} {} = {}",
                        metal_type,
                        decl.name,
                        self.convert_expression(init)
                    )
                } else {
                    format!("{} {}", metal_type, decl.name)
                }
            }
            Statement::Assign(assignment) => {
                format!(
                    "{} = {}",
                    self.convert_expression(&assignment.target),
                    self.convert_expression(&assignment.value)
                )
            }
            _ => self.convert_statement(stmt).trim().to_string(),
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
                format!(
                    "({} {} {})",
                    self.convert_expression(left),
                    self.convert_operator(op),
                    self.convert_expression(right)
                )
            }
            Expression::ThreadIdx(dim) => {
                format!("int(cuda_thread_idx.{})", self.dimension_component(dim))
            }
            Expression::BlockIdx(dim) => {
                format!("int(cuda_block_idx.{})", self.dimension_component(dim))
            }
            Expression::BlockDim(dim) => {
                format!("int(cuda_block_dim.{})", self.dimension_component(dim))
            }
            Expression::GridDim(dim) => {
                let component = self.dimension_component(dim);
                format!("int((cuda_threads_per_grid.{0} + cuda_block_dim.{0} - 1) / cuda_block_dim.{0})", component)
            }
            Expression::ArrayAccess { array, index } => {
                format!(
                    "{}[{}]",
                    self.convert_expression(array),
                    self.convert_expression(index)
                )
            }
            Expression::MathFunction { name, arguments } => {
                let metal_name = self.convert_math_function(name);
                let args = arguments
                    .iter()
                    .map(|arg| self.convert_expression(arg))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", metal_name, args)
            }
            Expression::Infinity => "INFINITY".to_string(),
            Expression::NegativeInfinity => "-INFINITY".to_string(),
            _ => "/* unsupported expression */".to_string(),
        }
    }

    fn dimension_component(&self, dim: &Dimension) -> &'static str {
        match dim {
            Dimension::X => "x",
            Dimension::Y => "y",
            Dimension::Z => "z",
        }
    }

    fn convert_operator(&self, op: &Operator) -> &str {
        match op {
            Operator::Add => "+",
            Operator::Subtract => "-",
            Operator::Multiply => "*",
            Operator::Divide => "/",
            Operator::Modulo => "%",
            Operator::LessThan => "<",
            Operator::LessThanEqual => "<=",
            Operator::GreaterThan => ">",
            Operator::GreaterThanEqual => ">=",
            Operator::Equal => "==",
            Operator::NotEqual => "!=",
            Operator::LogicalAnd => "&&",
            Operator::LogicalOr => "||",
            Operator::BitwiseAnd => "&",
            Operator::BitwiseOr => "|",
            Operator::BitwiseXor => "^",
            Operator::LeftShift => "<<",
            Operator::RightShift => ">>",
            _ => "+", // default for unsupported operators
        }
    }

    fn convert_math_function<'a>(&self, name: &'a str) -> &'a str {
        match name {
            "sin" => "sin",
            "cos" => "cos",
            "tan" => "tan",
            "exp" => "exp",
            "expf" => "exp", // CUDA expf -> Metal exp
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

        format!(
            r#"import Metal
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
"#,
            metal_runner_class,
            self.kernel_name,
            parameter_init,
            metal_shader,
            self.kernel_name,
            kernel_call
        )
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
    
    func executeKernel(inputs: [(data: Any, type: Any.Type)], outputType: Float.Type, outputBufferIndex: Int, dispatchWidth: Int, dispatchHeight: Int = 1) throws -> [Float] {
        guard !inputs.isEmpty else { throw MetalError.invalidInput }
        
        var buffers: [MTLBuffer] = []
        
        let problemSize = max(dispatchWidth * dispatchHeight, 1)
        
        // Allocate buffers
        for (index, input) in inputs.enumerated() {
            if let array = input.data as? [Float] {
                guard let buffer = device.makeBuffer(bytes: array,
                                                   length: MemoryLayout<Float>.stride * array.count,
                                                   options: .storageModeShared) else {
                    throw MetalError.bufferAllocationFailed
                }
                buffers.append(buffer)
            } else if let scalar = input.data as? Int32 {
                guard let buffer = device.makeBuffer(bytes: [scalar],
                                                   length: MemoryLayout<Int32>.size,
                                                   options: .storageModeShared) else {
                    throw MetalError.bufferAllocationFailed
                }
                buffers.append(buffer)
            } else if let scalar = input.data as? Int {
                let int32Value = Int32(scalar)
                guard let buffer = device.makeBuffer(bytes: [int32Value],
                                                   length: MemoryLayout<Int32>.size,
                                                   options: .storageModeShared) else {
                    throw MetalError.bufferAllocationFailed
                }
                buffers.append(buffer)
            } else if let scalar = input.data as? Float {
                guard let buffer = device.makeBuffer(bytes: [scalar],
                                                   length: MemoryLayout<Float>.size,
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
        
        let threadGroupSize: MTLSize
        let gridSize: MTLSize
        if dispatchHeight > 1 {
            threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
            gridSize = MTLSize(
                width: (dispatchWidth + threadGroupSize.width - 1) / threadGroupSize.width,
                height: (dispatchHeight + threadGroupSize.height - 1) / threadGroupSize.height,
                depth: 1
            )
        } else {
            let width = min(max(pipeline.maxTotalThreadsPerThreadgroup, 1), 256)
            threadGroupSize = MTLSize(width: width, height: 1, depth: 1)
            gridSize = MTLSize(width: (dispatchWidth + width - 1) / width, height: 1, depth: 1)
        }
        
        computeEncoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        if let error = commandBuffer.error {
            throw MetalError.executionFailed
        }
        
        guard outputBufferIndex >= 0 && outputBufferIndex < buffers.count else {
            throw MetalError.invalidInput
        }

        let outputBuffer = buffers[outputBufferIndex]
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
        let mut init = String::new();
        let dispatch_width = if self.dimensions == 2 {
            self.width.unwrap_or(32)
        } else {
            1024
        };
        let dispatch_height = if self.dimensions == 2 {
            self.height.unwrap_or(24)
        } else {
            1
        };

        init.push_str("// Initialize generated test data\n");
        init.push_str(&format!("let dispatchWidth = {}\n", dispatch_width));
        init.push_str(&format!("let dispatchHeight = {}\n", dispatch_height));
        init.push_str("let problemSize = dispatchWidth * dispatchHeight\n\n");

        for (index, param) in self.parameters.iter().enumerate() {
            match &param.param_type {
                Type::Pointer(_) if self.is_output_parameter(param, index) => {
                    init.push_str(&format!(
                        "var {} = Array(repeating: Float(0), count: problemSize)\n",
                        param.name
                    ));
                }
                Type::Pointer(_) => {
                    init.push_str(&format!(
                        "let {} = Array(0..<problemSize).map {{ Float($0 + {}) }}\n",
                        param.name, index
                    ));
                }
                Type::Int => {
                    let value = self.default_int_value(&param.name);
                    init.push_str(&format!("let {} = Int32({})\n", param.name, value));
                }
                Type::Float => {
                    init.push_str(&format!("let {} = Float(1.0)\n", param.name));
                }
                _ => {}
            }
        }

        init.push_str("\nprint(\"• Created generated inputs with \\(problemSize) elements\")");
        init
    }

    fn default_int_value(&self, name: &str) -> &'static str {
        let lower_name = name.to_ascii_lowercase();
        match lower_name.as_str() {
            "m" | "height" | "h" | "rows" => "dispatchHeight",
            "n" | "width" | "w" | "cols" => "dispatchWidth",
            _ => "problemSize",
        }
    }

    fn generate_kernel_call(&self) -> String {
        let inputs = self
            .parameters
            .iter()
            .map(|param| match &param.param_type {
                Type::Pointer(_) => format!("    (data: {}, type: [Float].self)", param.name),
                Type::Int => format!("    (data: {}, type: Int32.self)", param.name),
                Type::Float => format!("    (data: {}, type: Float.self)", param.name),
                _ => format!("    (data: {}, type: Float.self)", param.name),
            })
            .collect::<Vec<_>>()
            .join(",\n");

        let output_index = self.output_buffer_index().unwrap_or(0);

        format!(
            r#"let inputs: [(data: Any, type: Any.Type)] = [
{}
]

let result = try runner.executeKernel(
    inputs: inputs,
    outputType: Float.self,
    outputBufferIndex: {},
    dispatchWidth: dispatchWidth,
    dispatchHeight: dispatchHeight
)"#,
            inputs, output_index
        )
    }
}
