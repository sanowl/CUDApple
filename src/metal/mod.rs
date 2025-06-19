use host::MetalKernelConfig;

use crate::parser::unified_ast::{
    CudaProgram, Dimension, Expression, KernelFunction, Operator, Statement,
    Type,
};
use std::fmt::Write;
pub mod host;

#[derive(Debug)]
pub struct MetalShader {
    source: String,
    config: MetalKernelConfig,
}

impl MetalShader {
    pub fn new() -> Self {
        Self {
            source: String::new(),
            config: MetalKernelConfig {
                dimensions: 1, // Default to 1D
                grid_size: (4096, 1, 1),
                threadgroup_size: (256, 1, 1),
            },
        }
    }

    pub fn set_config(&mut self, config: MetalKernelConfig) {
        self.config = config;
    }

    pub fn generate(&mut self, program: &CudaProgram) -> Result<(), String> {
        // Generate Metal shader header with proper indentation
        writeln!(self.source, "            #include <metal_stdlib>").map_err(|e| e.to_string())?;
        writeln!(self.source, "            #include <metal_math>").map_err(|e| e.to_string())?;
        writeln!(self.source, "            using namespace metal;\n").map_err(|e| e.to_string())?;

        // Generate each kernel function
        for kernel in &program.device_code {
            self.generate_kernel_with_indent(kernel, "            ")?;
        }
        Ok(())
    }

    fn generate_kernel_with_indent(
        &mut self,
        kernel: &KernelFunction,
        indent: &str,
    ) -> Result<(), String> {
        // Write kernel signature with indentation
        writeln!(self.source, "{}kernel void {}(", indent, kernel.name)
            .map_err(|e| e.to_string())?;

        // Generate parameters with proper indentation
        let param_indent = format!("{}    ", indent);
        let mut first = true;
        for (index, param) in kernel.parameters.iter().enumerate() {
            if !first {
                writeln!(self.source, ",").map_err(|e| e.to_string())?;
            }
            first = false;

            write!(self.source, "{}", param_indent).map_err(|e| e.to_string())?;
            match &param.param_type {
                Type::Pointer(_) => {
                    write!(
                        self.source,
                        "device float* {} [[buffer({})]]",
                        param.name, index
                    )
                    .map_err(|e| e.to_string())?;
                }
                Type::Int => {
                    write!(
                        self.source,
                        "constant uint& {} [[buffer({})]]",
                        param.name, index
                    )
                    .map_err(|e| e.to_string())?;
                }
                _ => {
                    return Err(format!(
                        "Unsupported parameter type: {:?}",
                        param.param_type
                    ))
                }
            }
        }

        // Add thread position parameters
        if !first {
            writeln!(self.source, ",").map_err(|e| e.to_string())?;
        }
        match self.config.dimensions {
            1 => {
                write!(
                    self.source,
                    "{}uint32_t index [[thread_position_in_grid]]",
                    param_indent
                )
            }
            2 => {
                write!(
                    self.source,
                    "{}uint2 thread_position_in_grid [[thread_position_in_grid]]",
                    param_indent
                )
            }
            _ => return Err("Unsupported dimensions".to_string()),
        }
        .map_err(|e| e.to_string())?;

        writeln!(self.source, ") {{").map_err(|e| e.to_string())?;

        // Translate kernel body statements with proper indentation
        for stmt in &kernel.body.statements {
            self.translate_statement_with_indent(stmt, &format!("{}    ", indent))?;
        }

        // Close kernel function
        writeln!(self.source, "{}}}", indent).map_err(|e| e.to_string())?;
        writeln!(self.source).map_err(|e| e.to_string())?;

        Ok(())
    }

    fn translate_statement_with_indent(
        &mut self,
        stmt: &Statement,
        indent: &str,
    ) -> Result<(), String> {
        match stmt {
            Statement::VariableDecl(decl) => {
                write!(self.source, "{}", indent).map_err(|e| e.to_string())?;
                match decl.var_type {
                    Type::Int => write!(self.source, "int32_t").map_err(|e| e.to_string())?,
                    _ => write!(self.source, "{}", decl.var_type).map_err(|e| e.to_string())?,
                }
                write!(self.source, " {} = ", decl.name).map_err(|e| e.to_string())?;
                if let Some(init) = &decl.initializer {
                    self.translate_expression(init)?;
                }
                writeln!(self.source, ";").map_err(|e| e.to_string())?;
            }
            Statement::IfStmt { condition, body, .. } => {
                write!(self.source, "{}if (", indent).map_err(|e| e.to_string())?;
                self.translate_expression(condition)?;
                writeln!(self.source, ") {{").map_err(|e| e.to_string())?;

                // Translate if body with additional indentation
                for stmt in &body.statements {
                    self.translate_statement_with_indent(stmt, &format!("{}    ", indent))?;
                }

                writeln!(self.source, "{}}}", indent).map_err(|e| e.to_string())?;
            }
            Statement::Assign(assign) => {
                write!(self.source, "{}", indent).map_err(|e| e.to_string())?;
                self.translate_expression(&assign.target)?;
                write!(self.source, " = ").map_err(|e| e.to_string())?;
                self.translate_expression(&assign.value)?;
                writeln!(self.source, ";").map_err(|e| e.to_string())?;
            }
            _ => {
                // Handle unsupported statements gracefully
                write!(self.source, "{}// Unsupported statement\n", indent).map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }

    fn translate_expression(&mut self, expr: &Expression) -> Result<(), String> {
        match expr {
            Expression::ThreadIdx(dim) | Expression::BlockIdx(dim) => {
                match dim {
                    Dimension::X => write!(self.source, "index"),
                    Dimension::Y => write!(self.source, "thread_position_in_grid.y"),
                    _ => return Err("Unsupported thread index dimension".to_string()),
                }
                .map_err(|e| e.to_string())?;
            }
            Expression::ArrayAccess { array, index } => {
                self.translate_expression(array)?;
                write!(self.source, "[").map_err(|e| e.to_string())?;
                self.translate_expression(index)?;
                write!(self.source, "]").map_err(|e| e.to_string())?;
            }
            Expression::BinaryOp(lhs, op, rhs) => {
                self.translate_expression(lhs)?;
                write!(self.source, " {} ", Self::operator_to_string(op))
                    .map_err(|e| e.to_string())?;
                self.translate_expression(rhs)?;
            }
            Expression::Variable(name) => {
                write!(self.source, "{}", name).map_err(|e| e.to_string())?;
            }
            Expression::IntegerLiteral(value) => {
                write!(self.source, "{}", value).map_err(|e| e.to_string())?;
            }
            Expression::FloatLiteral(value) => {
                write!(self.source, "{}f", value).map_err(|e| e.to_string())?;
            }
            Expression::MathFunction { name, arguments } => {
                write!(self.source, "{}(", name).map_err(|e| e.to_string())?;
                for (i, arg) in arguments.iter().enumerate() {
                    if i > 0 {
                        write!(self.source, ", ").map_err(|e| e.to_string())?;
                    }
                    self.translate_expression(arg)?;
                }
                write!(self.source, ")").map_err(|e| e.to_string())?;
            }
            _ => {
                write!(self.source, "/* unsupported expression */").map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }

    fn operator_to_string(op: &Operator) -> &'static str {
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
            _ => "+", // default for unsupported operators
        }
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    fn translate_type(&self, t: &Type) -> String {
        match t {
            Type::Int => "int".to_string(),
            Type::Float => "float".to_string(),
            Type::Void => "void".to_string(),
            Type::Pointer(inner) => self.translate_type(inner),
            _ => "float".to_string(), // default for unsupported types
        }
    }
}

fn is_thread_index_component(expr: &Expression) -> bool {
    matches!(expr, Expression::ThreadIdx(_) | Expression::BlockIdx(_))
}

fn is_x_component(expr: &Expression) -> bool {
    matches!(expr, Expression::ThreadIdx(Dimension::X) | Expression::BlockIdx(Dimension::X))
}

fn is_y_component(expr: &Expression) -> bool {
    matches!(expr, Expression::ThreadIdx(Dimension::Y) | Expression::BlockIdx(Dimension::Y))
}
