use crate::parser::unified_ast::*;
use std::collections::{HashMap, HashSet};

pub struct CodeAnalyzer {
    pub kernel_complexity: HashMap<String, KernelComplexity>,
    pub memory_access_patterns: HashMap<String, MemoryAccessPattern>,
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
}

#[derive(Debug, Clone)]
pub struct KernelComplexity {
    pub dimensions: usize,
    pub estimated_threads_per_block: usize,
    pub shared_memory_usage: usize,
    pub register_pressure: RegisterPressure,
    pub divergence_risk: DivergenceRisk,
    pub math_operations: MathOperationProfile,
}

#[derive(Debug, Clone)]
pub struct MemoryAccessPattern {
    pub global_reads: Vec<MemoryAccess>,
    pub global_writes: Vec<MemoryAccess>,
    pub shared_memory_usage: Vec<SharedMemoryUsage>,
    pub coalescing_efficiency: CoalescingEfficiency,
    pub bank_conflicts: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryAccess {
    pub base_expression: Expression,
    pub index_pattern: IndexPattern,
    pub access_type: AccessType,
    pub estimated_frequency: f64,
}

#[derive(Debug, Clone)]
pub enum IndexPattern {
    Sequential,     // arr[i], arr[i+1], arr[i+2]...
    Strided(isize), // arr[i], arr[i+stride], arr[i+2*stride]...
    Random,         // Unpredictable access pattern
    MatrixRow,      // arr[row*width + col]
    MatrixColumn,   // arr[col*height + row]
    Convolution,    // Complex 2D/3D patterns
}

#[derive(Debug, Clone)]
pub enum AccessType {
    Read,
    Write,
    ReadWrite,
    Atomic,
}

#[derive(Debug, Clone)]
pub struct SharedMemoryUsage {
    pub declaration: Declaration,
    pub access_patterns: Vec<MemoryAccess>,
    pub synchronization_points: Vec<SyncPoint>,
}

#[derive(Debug, Clone)]
pub enum SyncPoint {
    ThreadSync,
    WarpSync,
    BlockSync,
}

#[derive(Debug, Clone)]
pub enum CoalescingEfficiency {
    Excellent,  // 100% coalesced
    Good,       // 75-99% coalesced
    Fair,       // 50-74% coalesced
    Poor,       // 25-49% coalesced
    Terrible,   // <25% coalesced
}

#[derive(Debug, Clone)]
pub enum RegisterPressure {
    Low,    // <32 registers
    Medium, // 32-64 registers
    High,   // 64-96 registers
    Extreme, // >96 registers
}

#[derive(Debug, Clone)]
pub enum DivergenceRisk {
    None,   // All threads follow same path
    Low,    // Minimal branching
    Medium, // Some conditional execution
    High,   // Complex branching patterns
}

#[derive(Debug, Clone)]
pub struct MathOperationProfile {
    pub float_ops: usize,
    pub int_ops: usize,
    pub transcendental_ops: usize,
    pub special_functions: Vec<String>,
    pub estimated_arithmetic_intensity: f64,
}

#[derive(Debug, Clone)]
pub enum OptimizationOpportunity {
    VectorizeLoops {
        loop_location: String,
        estimated_speedup: f64,
    },
    ImproveMemoryCoalescing {
        access_location: String,
        suggested_change: String,
    },
    ReduceSharedMemoryBankConflicts {
        conflict_location: String,
        suggested_padding: usize,
    },
    OptimizeMathFunctions {
        function_name: String,
        suggested_alternative: String,
    },
    ParallelizeSequentialCode {
        code_location: String,
        parallelization_strategy: String,
    },
}

impl CodeAnalyzer {
    pub fn new() -> Self {
        Self {
            kernel_complexity: HashMap::new(),
            memory_access_patterns: HashMap::new(),
            optimization_opportunities: Vec::new(),
        }
    }

    pub fn analyze_program(&mut self, program: &CudaProgram) -> AnalysisResult {
        let mut results = AnalysisResult {
            overall_score: 0.0,
            performance_bottlenecks: Vec::new(),
            metal_compatibility: MetalCompatibility::High,
            estimated_performance_gain: 0.0,
        };

        for kernel in &program.device_code {
            let complexity = self.analyze_kernel_complexity(kernel);
            let memory_pattern = self.analyze_memory_patterns(kernel);
            
            self.kernel_complexity.insert(kernel.name.clone(), complexity);
            self.memory_access_patterns.insert(kernel.name.clone(), memory_pattern);
            
            self.identify_optimization_opportunities(kernel);
        }

        self.calculate_overall_metrics(&mut results);
        results
    }

    fn analyze_kernel_complexity(&self, kernel: &KernelFunction) -> KernelComplexity {
        let dimensions = self.determine_dimensions(kernel);
        let register_pressure = self.estimate_register_pressure(kernel);
        let divergence_risk = self.analyze_divergence_risk(kernel);
        let math_ops = self.profile_math_operations(kernel);

        KernelComplexity {
            dimensions,
            estimated_threads_per_block: self.estimate_optimal_block_size(kernel),
            shared_memory_usage: self.calculate_shared_memory_usage(kernel),
            register_pressure,
            divergence_risk,
            math_operations: math_ops,
        }
    }

    fn determine_dimensions(&self, kernel: &KernelFunction) -> usize {
        let mut uses_y = false;
        let mut uses_z = false;

        self.analyze_expressions_for_dimensions(&kernel.body, &mut uses_y, &mut uses_z);

        if uses_z { 3 } else if uses_y { 2 } else { 1 }
    }

    fn analyze_expressions_for_dimensions(&self, block: &Block, uses_y: &mut bool, uses_z: &mut bool) {
        for stmt in &block.statements {
            self.analyze_statement_for_dimensions(stmt, uses_y, uses_z);
        }
    }

    fn analyze_statement_for_dimensions(&self, stmt: &Statement, uses_y: &mut bool, uses_z: &mut bool) {
        match stmt {
            Statement::VariableDecl(decl) => {
                if let Some(init) = &decl.initializer {
                    self.analyze_expression_for_dimensions(init, uses_y, uses_z);
                }
            }
            Statement::Assign(assign) => {
                self.analyze_expression_for_dimensions(&assign.target, uses_y, uses_z);
                self.analyze_expression_for_dimensions(&assign.value, uses_y, uses_z);
            }
            Statement::IfStmt { condition, body, else_body } => {
                self.analyze_expression_for_dimensions(condition, uses_y, uses_z);
                self.analyze_expressions_for_dimensions(body, uses_y, uses_z);
                if let Some(else_body) = else_body {
                    self.analyze_expressions_for_dimensions(else_body, uses_y, uses_z);
                }
            }
            Statement::ForLoop { init, condition, increment, body } => {
                self.analyze_statement_for_dimensions(init, uses_y, uses_z);
                self.analyze_expression_for_dimensions(condition, uses_y, uses_z);
                self.analyze_statement_for_dimensions(increment, uses_y, uses_z);
                self.analyze_expressions_for_dimensions(body, uses_y, uses_z);
            }
            _ => {}
        }
    }

    fn analyze_expression_for_dimensions(&self, expr: &Expression, uses_y: &mut bool, uses_z: &mut bool) {
        match expr {
            Expression::ThreadIdx(Dimension::Y) | Expression::BlockIdx(Dimension::Y) | Expression::BlockDim(Dimension::Y) => {
                *uses_y = true;
            }
            Expression::ThreadIdx(Dimension::Z) | Expression::BlockIdx(Dimension::Z) | Expression::BlockDim(Dimension::Z) => {
                *uses_z = true;
            }
            Expression::BinaryOp(lhs, _, rhs) => {
                self.analyze_expression_for_dimensions(lhs, uses_y, uses_z);
                self.analyze_expression_for_dimensions(rhs, uses_y, uses_z);
            }
            Expression::ArrayAccess { array, index } => {
                self.analyze_expression_for_dimensions(array, uses_y, uses_z);
                self.analyze_expression_for_dimensions(index, uses_y, uses_z);
            }
            _ => {}
        }
    }

    fn estimate_register_pressure(&self, _kernel: &KernelFunction) -> RegisterPressure {
        // Simplified analysis - in production this would be much more sophisticated
        RegisterPressure::Medium
    }

    fn analyze_divergence_risk(&self, kernel: &KernelFunction) -> DivergenceRisk {
        let mut branch_count = 0;
        self.count_branches(&kernel.body, &mut branch_count);
        
        match branch_count {
            0 => DivergenceRisk::None,
            1..=2 => DivergenceRisk::Low,
            3..=5 => DivergenceRisk::Medium,
            _ => DivergenceRisk::High,
        }
    }

    fn count_branches(&self, block: &Block, count: &mut usize) {
        for stmt in &block.statements {
            match stmt {
                Statement::IfStmt { body, else_body, .. } => {
                    *count += 1;
                    self.count_branches(body, count);
                    if let Some(else_body) = else_body {
                        self.count_branches(else_body, count);
                    }
                }
                Statement::ForLoop { body, .. } => {
                    self.count_branches(body, count);
                }
                _ => {}
            }
        }
    }

    fn profile_math_operations(&self, kernel: &KernelFunction) -> MathOperationProfile {
        let mut profile = MathOperationProfile {
            float_ops: 0,
            int_ops: 0,
            transcendental_ops: 0,
            special_functions: Vec::new(),
            estimated_arithmetic_intensity: 0.0,
        };

        self.count_operations(&kernel.body, &mut profile);
        
        // Calculate arithmetic intensity (ops per memory access)
        let total_ops = profile.float_ops + profile.int_ops + profile.transcendental_ops;
        profile.estimated_arithmetic_intensity = total_ops as f64 / 10.0; // Rough estimate

        profile
    }

    fn count_operations(&self, block: &Block, profile: &mut MathOperationProfile) {
        for stmt in &block.statements {
            match stmt {
                Statement::Assign(assign) => {
                    self.count_expression_operations(&assign.value, profile);
                }
                Statement::IfStmt { condition, body, else_body } => {
                    self.count_expression_operations(condition, profile);
                    self.count_operations(body, profile);
                    if let Some(else_body) = else_body {
                        self.count_operations(else_body, profile);
                    }
                }
                Statement::ForLoop { condition, body, .. } => {
                    self.count_expression_operations(condition, profile);
                    self.count_operations(body, profile);
                }
                _ => {}
            }
        }
    }

    fn count_expression_operations(&self, expr: &Expression, profile: &mut MathOperationProfile) {
        match expr {
            Expression::BinaryOp(lhs, op, rhs) => {
                match op {
                    Operator::Add | Operator::Subtract | Operator::Multiply | Operator::Divide => {
                        profile.float_ops += 1;
                    }
                    _ => profile.int_ops += 1,
                }
                self.count_expression_operations(lhs, profile);
                self.count_expression_operations(rhs, profile);
            }
            Expression::MathFunction { name, arguments } => {
                match name.as_str() {
                    "sin" | "cos" | "tan" | "exp" | "log" | "sqrt" | "pow" => {
                        profile.transcendental_ops += 1;
                        profile.special_functions.push(name.clone());
                    }
                    _ => profile.float_ops += 1,
                }
                for arg in arguments {
                    self.count_expression_operations(arg, profile);
                }
            }
            Expression::ArrayAccess { array, index } => {
                self.count_expression_operations(array, profile);
                self.count_expression_operations(index, profile);
            }
            _ => {}
        }
    }

    fn estimate_optimal_block_size(&self, _kernel: &KernelFunction) -> usize {
        // Smart block size estimation based on kernel characteristics
        256 // Default to 256 for now
    }

    fn calculate_shared_memory_usage(&self, kernel: &KernelFunction) -> usize {
        let mut shared_memory = 0;
        
        for stmt in &kernel.body.statements {
            if let Statement::VariableDecl(decl) = stmt {
                if matches!(decl.memory_space, MemorySpace::Shared) {
                    shared_memory += self.estimate_type_size(&decl.var_type);
                }
            }
        }
        
        shared_memory
    }

    fn estimate_type_size(&self, t: &Type) -> usize {
        match t {
            Type::Int | Type::UInt | Type::Float => 4,
            Type::Long | Type::ULong | Type::Double => 8,
            Type::Short | Type::UShort | Type::Half => 2,
            Type::Char | Type::UChar | Type::Bool => 1,
            Type::Vector(base, size) => self.estimate_type_size(base) * size,
            Type::Array(base, Some(size)) => self.estimate_type_size(base) * size,
            _ => 4, // Default
        }
    }

    fn analyze_memory_patterns(&self, _kernel: &KernelFunction) -> MemoryAccessPattern {
        // Simplified for now - would analyze actual memory access patterns
        MemoryAccessPattern {
            global_reads: Vec::new(),
            global_writes: Vec::new(),
            shared_memory_usage: Vec::new(),
            coalescing_efficiency: CoalescingEfficiency::Good,
            bank_conflicts: 0,
        }
    }

    fn identify_optimization_opportunities(&mut self, _kernel: &KernelFunction) {
        // Analyze and suggest optimizations
        // This would be much more sophisticated in production
    }

    fn calculate_overall_metrics(&self, results: &mut AnalysisResult) {
        results.overall_score = 85.0; // Placeholder
        results.estimated_performance_gain = 3.2; // Placeholder
    }
}

#[derive(Debug)]
pub struct AnalysisResult {
    pub overall_score: f64,
    pub performance_bottlenecks: Vec<String>,
    pub metal_compatibility: MetalCompatibility,
    pub estimated_performance_gain: f64,
}

#[derive(Debug)]
pub enum MetalCompatibility {
    High,
    Medium,
    Low,
    RequiresChanges,
} 