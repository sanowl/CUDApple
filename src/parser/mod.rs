pub mod unified_ast;
pub mod cuda_parser;
pub mod metal_generator;

pub use unified_ast::*;

use anyhow::Result;

/// High-level function to parse CUDA source code into a CudaProgram
pub fn parse_cuda(source: &str) -> Result<CudaProgram> {
    // Try to parse as a single kernel function first
    match cuda_parser::cuda_parser::kernel_function(source.trim()) {
        Ok(kernel) => {
            Ok(CudaProgram {
                device_code: vec![kernel],
                host_code: Vec::new(),
                type_definitions: Vec::new(),
                constants: Vec::new(),
                textures: Vec::new(),
            })
        }
        Err(e) => {
            // If single kernel parsing fails, try to parse multiple kernels
            let mut kernels = Vec::new();
            let lines: Vec<&str> = source.lines().collect();
            let mut current_kernel = String::new();
            let mut in_kernel = false;
            let mut brace_count = 0;
            
            for line in lines {
                let trimmed = line.trim();
                
                // Skip empty lines and comments
                if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with("/*") {
                    continue;
                }
                
                // Start of a kernel function
                if trimmed.contains("__global__") {
                    in_kernel = true;
                    current_kernel.clear();
                    current_kernel.push_str(line);
                    current_kernel.push('\n');
                    brace_count = 0;
                    continue;
                }
                
                if in_kernel {
                    current_kernel.push_str(line);
                    current_kernel.push('\n');
                    
                    // Count braces to know when kernel ends
                    for ch in line.chars() {
                        match ch {
                            '{' => brace_count += 1,
                            '}' => {
                                brace_count -= 1;
                                if brace_count == 0 {
                                    // End of kernel - try to parse it
                                    match cuda_parser::cuda_parser::kernel_function(current_kernel.trim()) {
                                        Ok(kernel) => kernels.push(kernel),
                                        Err(kernel_err) => {
                                            log::warn!("Failed to parse kernel: {}", kernel_err);
                                            // Continue trying to parse other kernels
                                        }
                                    }
                                    in_kernel = false;
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            
            if kernels.is_empty() {
                anyhow::bail!("No valid kernels found. Parse error: {}", e);
            }
            
            Ok(CudaProgram {
                device_code: kernels,
                host_code: Vec::new(),
                type_definitions: Vec::new(),
                constants: Vec::new(),
                textures: Vec::new(),
            })
        }
    }
}