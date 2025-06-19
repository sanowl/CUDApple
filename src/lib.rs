pub mod parser;
pub mod metal;
pub mod analyzer;
pub mod optimizer;

pub use parser::unified_ast::*;
pub use analyzer::{CodeAnalyzer, AnalysisResult, MetalCompatibility};
pub use optimizer::{KernelOptimizer, OptimizationResult};

use anyhow::Result;

/// High-level interface for CUDA to Metal transpilation with intelligent optimization
pub struct CudaAppleTranspiler {
    analyzer: CodeAnalyzer,
    optimizer: KernelOptimizer,
    config: TranspilerConfig,
}

#[derive(Debug, Clone)]
pub struct TranspilerConfig {
    pub optimization_level: u8,
    pub target_device: TargetDevice,
    pub enable_fast_math: bool,
    pub enable_vectorization: bool,
    pub thread_group_size_hint: Option<(usize, usize, usize)>,
}

#[derive(Debug, Clone)]
pub enum TargetDevice {
    AppleSilicon,
    IntelMac,
    Generic,
}

impl Default for TranspilerConfig {
    fn default() -> Self {
        Self {
            optimization_level: 2,
            target_device: TargetDevice::AppleSilicon,
            enable_fast_math: true,
            enable_vectorization: true,
            thread_group_size_hint: None,
        }
    }
}

impl CudaAppleTranspiler {
    pub fn new() -> Self {
        Self::with_config(TranspilerConfig::default())
    }

    pub fn with_config(config: TranspilerConfig) -> Self {
        Self {
            analyzer: CodeAnalyzer::new(),
            optimizer: KernelOptimizer::new(),
            config,
        }
    }

    /// Parse CUDA source code and generate optimized Metal code
    pub fn transpile(&mut self, cuda_source: &str) -> Result<TranspilationResult> {
        // Parse CUDA code
        let mut cuda_program = parser::parse_cuda(cuda_source)?;

        // Analyze code for optimization opportunities
        let analysis = if self.config.optimization_level > 0 {
            Some(self.analyzer.analyze_program(&cuda_program))
        } else {
            None
        };

        // Apply optimizations
        let optimization_result = if self.config.optimization_level > 0 {
            Some(self.optimizer.optimize_program(&mut cuda_program, analysis.as_ref().unwrap()))
        } else {
            None
        };

        // Generate Metal code
        let metal_code = self.generate_metal_code(&cuda_program)?;
        let swift_runner = self.generate_swift_runner(&cuda_program)?;

        Ok(TranspilationResult {
            metal_shader: metal_code,
            swift_runner,
            cuda_program,
            analysis_result: analysis,
            optimization_result,
        })
    }

    fn generate_metal_code(&self, program: &CudaProgram) -> Result<String> {
        if program.device_code.is_empty() {
            anyhow::bail!("No kernels found in CUDA program");
        }

        let generator = parser::metal_generator::MetalCodeGenerator::new(&program.device_code[0]);
        Ok(generator.generate_metal_shader(&program.device_code[0]))
    }

    fn generate_swift_runner(&self, program: &CudaProgram) -> Result<String> {
        if program.device_code.is_empty() {
            anyhow::bail!("No kernels found in CUDA program");
        }

        let generator = parser::metal_generator::MetalCodeGenerator::new(&program.device_code[0]);
        Ok(generator.generate_swift_runner(&program.device_code[0]))
    }

    /// Get performance analysis for CUDA code without transpilation
    pub fn analyze_only(&mut self, cuda_source: &str) -> Result<AnalysisResult> {
        let cuda_program = parser::parse_cuda(cuda_source)?;
        Ok(self.analyzer.analyze_program(&cuda_program))
    }

    /// Generate performance optimization report
    pub fn generate_optimization_report(&self) -> String {
        self.optimizer.generate_optimization_report()
    }
}

#[derive(Debug)]
pub struct TranspilationResult {
    pub metal_shader: String,
    pub swift_runner: String,
    pub cuda_program: CudaProgram,
    pub analysis_result: Option<AnalysisResult>,
    pub optimization_result: Option<OptimizationResult>,
}

impl Default for CudaAppleTranspiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for quick transpilation
pub fn transpile_cuda_to_metal(cuda_source: &str) -> Result<String> {
    let mut transpiler = CudaAppleTranspiler::new();
    let result = transpiler.transpile(cuda_source)?;
    Ok(result.metal_shader)
}

/// Convenience function for analysis only
pub fn analyze_cuda_performance(cuda_source: &str) -> Result<AnalysisResult> {
    let mut transpiler = CudaAppleTranspiler::new();
    transpiler.analyze_only(cuda_source)
} 