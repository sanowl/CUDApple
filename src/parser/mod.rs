pub mod unified_ast;
pub mod cuda_parser;

pub use unified_ast::*;
pub use cuda_parser::*;

#[cfg(test)]
mod tests;

use crate::parser::cuda_grammar::cuda_parser;
use crate::parser::unified_ast::CudaProgram;
use anyhow::Result;

pub fn parse_cuda(source: &str) -> Result<CudaProgram> {
    let kernel = cuda_parser::kernel_function(source)
        .map_err(|e| anyhow::anyhow!("Failed to parse kernel: {}", e.to_string()))?;

    Ok(CudaProgram {
        device_code: vec![kernel],
    })
}
