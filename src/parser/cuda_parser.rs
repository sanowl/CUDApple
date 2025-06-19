use crate::parser::unified_ast::{
    AtomicOp, Assignment, Block, Declaration, Dimension, Expression,
    KernelFunction, MemoryFence, MemorySpace, Operator, Parameter, Qualifier,
    Statement, SyncScope, Type
};

peg::parser! {
    pub grammar cuda_parser() for str {
        rule _() = quiet!{([' ' | '\t' | '\n' | '\r'] / comment())*}

        rule comment() = block_comment() / line_comment()
        rule block_comment() = "/*" (!"*/" [_])* "*/"
        rule line_comment() = "//" (!"\n" [_])* ("\n" / ![_])

        rule memory_space() -> MemorySpace
            = "__shared__" { MemorySpace::Shared }
            / "__constant__" { MemorySpace::Constant }
            / "__device__" { MemorySpace::Default }
            / { MemorySpace::Global }

        rule vector_type() -> Type
            = base:$("float" / "int" / "double") size:$(['1'..='4']) {
                let base_type = match base {
                    "float" => Type::Float,
                    "int" => Type::Int,
                    "double" => Type::Float, // Map double to float for Metal
                    _ => unreachable!()
                };
                Type::Vector(Box::new(base_type), size.parse().unwrap())
            }

        rule atomic_operation() -> Statement
            = "atomicAdd" "(" target:expression() "," value:expression() ")" {
                Statement::AtomicOperation {
                    operation: AtomicOp::Add,
                    target,
                    value,
                    compare: None,
                }
            }
            / "atomicSub" "(" target:expression() "," value:expression() ")" {
                Statement::AtomicOperation {
                    operation: AtomicOp::Sub,
                    target,
                    value,
                    compare: None,
                }
            }

        pub rule kernel_function() -> KernelFunction
            = _ "__global__" _ "void" _ name:identifier() _ "(" _ params:parameter_list()? _ ")" _ body:block() {
                KernelFunction {
                    name,
                    parameters: params.unwrap_or_default(),
                    body,
                    launch_bounds: None,
                    shared_memory_size: None,
                    qualifiers: vec![],
                }
            }

        rule identifier() -> String
            = id:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '0'..='9' | '_']*) {
                id.to_string()
            }

        rule qualifier() -> Qualifier
            = "__restrict__" { Qualifier::Restrict }
            / { Qualifier::None }

        rule parameter() -> Parameter
            = param_type:type_specifier() _ "*" _ qualifier:qualifier() _ name:identifier() {
                Parameter {
                    param_type: Type::Pointer(Box::new(param_type)),
                    name,
                    qualifier
                }
            }
            / param_type:type_specifier() _ name:identifier() {
                Parameter {
                    param_type,
                    name,
                    qualifier: Qualifier::None
                }
            }

        rule parameter_list() -> Vec<Parameter>
            = first:parameter() rest:(_ "," _ p:parameter() { p })* {
                let mut params = vec![first];
                params.extend(rest);
                params
            }

        rule block() -> Block
            = _ "{" _ statements:(statement() ** _) _ "}" _ {
                Block { statements }
            }

        rule statement() -> Statement
            = for_loop()
            / variable_declaration()
            / shared_memory_declaration()
            / compound_assignment()
            / assignment()
            / if_statement()
            / device_function()
            / struct_declaration()
            / sync_threads()
            / atomic_operation()
            / memory_fence()
            / memory_barrier()
            / e:expression() _ ";" { Statement::Expression(Box::new(e)) }

        // Enhanced: Handle extern __shared__ declarations
        rule shared_memory_declaration() -> Statement
            = "extern" _ "__shared__" _ var_type:type_specifier() _ name:identifier() _ "[" _ "]" _ ";" {
                Statement::VariableDecl(Declaration {
                    var_type: Type::Array(Box::new(var_type), None),
                    name,
                    initializer: None,
                    memory_space: MemorySpace::Shared,
                    qualifiers: vec![],
                })
            }

        rule variable_declaration() -> Statement
            = mem:memory_space()? _ var_type:type_specifier() _ name:identifier() _ ptr:"*"? _ array_size:("[" _ size:expression() _ "]" { size })? _ init:("=" _ e:expression() { e })? _ ";" {
                let final_type = if let Some(_) = ptr {
                    Type::Pointer(Box::new(var_type))
                } else if let Some(size_expr) = array_size {
                    // For simplicity, assume integer literal array sizes
                    let size = match size_expr {
                        Expression::IntegerLiteral(n) => Some(n as usize),
                        _ => None,
                    };
                    Type::Array(Box::new(var_type), size)
                } else {
                    var_type
                };
                
                Statement::VariableDecl(Declaration {
                    var_type: final_type,
                    name,
                    initializer: init,
                    memory_space: mem.unwrap_or(MemorySpace::Default),
                    qualifiers: vec![],
                })
            }

        rule type_specifier() -> Type
            = "int" { Type::Int }
            / "float" { Type::Float }
            / "void" { Type::Void }
            / "bool" { Type::Int } // Map bool to int for Metal
            / vector_type()

        rule assignment() -> Statement
            = target:(array_access() / variable()) _ "=" _ value:expression() _ ";" {
                Statement::Assign(Assignment {
                    target,
                    value
                })
            }

        rule if_statement() -> Statement
            = "if" _ "(" _ condition:expression() _ ")" _ body:block() _ else_clause:("else" _ else_body:block() { else_body })? {
                Statement::IfStmt {
                    condition,
                    body,
                    else_body: else_clause,
                }
            }

        rule for_loop() -> Statement
            = "for" _ "(" _
              init:for_init() _ ";" _
              condition:expression() _ ";" _
              increment:for_increment() _
              ")" _
              body:block() {
                Statement::ForLoop {
                    init: Box::new(init),
                    condition,
                    increment: Box::new(increment),
                    body,
                }
            }

        rule for_init() -> Statement
            = var_type:type_specifier() _ name:identifier() _ "=" _ value:expression() {
                Statement::VariableDecl(Declaration {
                    var_type,
                    name,
                    initializer: Some(value),
                    memory_space: MemorySpace::Default,
                    qualifiers: vec![],
                })
            }

        rule for_increment() -> Statement
            = target:identifier() _ "++" {
                Statement::Assign(Assignment {
                    target: Expression::Variable(target.clone()),
                    value: Expression::BinaryOp(
                        Box::new(Expression::Variable(target)),
                        Operator::Add,
                        Box::new(Expression::IntegerLiteral(1))
                    )
                })
            }
            / target:identifier() _ "=" _ target2:identifier() _ "+" _ value:expression() {
                Statement::Assign(Assignment {
                    target: Expression::Variable(target),
                    value: Expression::BinaryOp(
                        Box::new(Expression::Variable(target2)),
                        Operator::Add,
                        Box::new(value)
                    )
                })
            }
            / target:identifier() _ "+=" _ value:expression() {
                Statement::CompoundAssign {
                    target: Expression::Variable(target),
                    operator: Operator::Add,
                    value
                }
            }

        rule array_access() -> Expression
            = array:identifier() _ "[" _ index:expression() _ "]" {
                Expression::ArrayAccess {
                    array: Box::new(Expression::Variable(array)),
                    index: Box::new(index)
                }
            }

        rule variable() -> Expression
            = name:identifier() { Expression::Variable(name) }

        rule math_function() -> Expression
            = "sin" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction { name: "sin".to_string(), arguments: vec![x] }
            }
            / "cos" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction { name: "cos".to_string(), arguments: vec![x] }
            }
            / "tan" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction { name: "tan".to_string(), arguments: vec![x] }
            }
            / "exp" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction { name: "exp".to_string(), arguments: vec![x] }
            }
            / "log" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction { name: "log".to_string(), arguments: vec![x] }
            }
            / "sqrt" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction { name: "sqrt".to_string(), arguments: vec![x] }
            }
            / "pow" _ "(" _ x:expression() _ "," _ y:expression() _ ")" {
                Expression::MathFunction { name: "pow".to_string(), arguments: vec![x, y] }
            }
            / "max" _ "(" _ x:expression() _ "," _ y:expression() _ ")" {
                Expression::MathFunction { name: "max".to_string(), arguments: vec![x, y] }
            }
            / "min" _ "(" _ x:expression() _ "," _ y:expression() _ ")" {
                Expression::MathFunction { name: "min".to_string(), arguments: vec![x, y] }
            }
            / "abs" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction { name: "abs".to_string(), arguments: vec![x] }
            }
            / "floor" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction { name: "floor".to_string(), arguments: vec![x] }
            }
            / "ceil" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction { name: "ceil".to_string(), arguments: vec![x] }
            }
            / "round" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction { name: "round".to_string(), arguments: vec![x] }
            }

        rule infinity() -> Expression
            = "INFINITY" { Expression::Infinity }
            / "-INFINITY" { Expression::NegativeInfinity }

        // Enhanced expression parsing with proper precedence
        rule expression() -> Expression = precedence! {
            x:(@) _ "&&" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::LogicalAnd, Box::new(y)) }
            x:(@) _ "||" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::LogicalOr, Box::new(y)) }
            --
            x:(@) _ "<" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::LessThan, Box::new(y)) }
            x:(@) _ ">=" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::GreaterThanEqual, Box::new(y)) }
            x:(@) _ "<=" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::LessThanEqual, Box::new(y)) }
            x:(@) _ ">" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::GreaterThan, Box::new(y)) }
            x:(@) _ "==" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Equal, Box::new(y)) }
            x:(@) _ "!=" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::NotEqual, Box::new(y)) }
            --
            x:(@) _ "+" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Add, Box::new(y)) }
            x:(@) _ "-" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Subtract, Box::new(y)) }
            --
            x:(@) _ "*" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Multiply, Box::new(y)) }
            x:(@) _ "/" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Divide, Box::new(y)) }
            x:(@) _ "%" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Modulo, Box::new(y)) }
            --
            n:number() { n }
            i:infinity() { i }
            t:thread_index() { t }
            m:math_function() { m }
            a:array_access() { a }
            v:variable() { v }
            "(" _ e:expression() _ ")" { e }
        }

        rule number() -> Expression
            = n:$(['0'..='9']+ "." ['0'..='9']* "f"?) {
                let n = n.trim_end_matches('f');
                Expression::FloatLiteral(n.parse::<f64>().unwrap())
            }
            / n:$(['0'..='9']+ "f") {
                let n = n.trim_end_matches('f');
                Expression::FloatLiteral(n.parse::<f64>().unwrap())
            }
            / n:$(['0'..='9']+) {
                Expression::IntegerLiteral(n.parse().unwrap())
            }

        rule thread_index() -> Expression
            = "threadIdx." d:dimension() { Expression::ThreadIdx(d) }
            / "blockIdx." d:dimension() { Expression::BlockIdx(d) }
            / "blockDim." d:dimension() { Expression::BlockDim(d) }

        rule dimension() -> Dimension
            = "x" { Dimension::X }
            / "y" { Dimension::Y }
            / "z" { Dimension::Z }

        rule compound_assignment() -> Statement
            = target:(array_access() / variable()) _ op:compound_operator() _ value:expression() _ ";" {
                Statement::CompoundAssign {
                    target,
                    operator: op,
                    value
                }
            }

        rule compound_operator() -> Operator
            = "+=" { Operator::Add }
            / "-=" { Operator::Subtract }
            / "*=" { Operator::Multiply }
            / "/=" { Operator::Divide }

        rule declaration() -> Declaration
            = mem:memory_space()? _ var_type:type_specifier() _ name:identifier() _ ";" {
                Declaration {
                    var_type,
                    name,
                    initializer: None,
                    memory_space: mem.unwrap_or(MemorySpace::Default),
                    qualifiers: vec![],
                }
            }

        rule struct_declaration() -> Statement
            = "struct" _ name:identifier() _ "{" _
              fields:(declaration() ** _) _
              "}" _ ";" {
                Statement::StructDecl {
                    name,
                    fields,
                }
            }

        rule device_function() -> Statement
            = "__device__" _ return_type:type_specifier() _
              name:identifier() _ "(" _ params:parameter_list()? _ ")" _ body:block() {
                Statement::DeviceFunction {
                    name,
                    parameters: params.unwrap_or_default(),
                    return_type,
                    body,
                    qualifiers: vec![],
                }
            }

        rule sync_threads() -> Statement
            = "__syncthreads" _ "(" _ ")" _ ";" { Statement::SyncThreads }

        rule memory_fence() -> Statement
            = "__threadfence()" _ ";" { 
                Statement::MemoryFence(MemoryFence::System) 
            }
            / "__threadfence_block()" _ ";" { 
                Statement::MemoryFence(MemoryFence::Shared) 
            }
            / "__threadfence_system()" _ ";" { 
                Statement::MemoryFence(MemoryFence::System) 
            }

        rule memory_barrier() -> Statement
            = "__syncthreads_and_fence" _ "(" _ scope:sync_scope() _ ")" _ ";" {
                Statement::MemoryBarrier {
                    scope,
                    fence: MemoryFence::System,
                }
            }

        rule sync_scope() -> SyncScope
            = "gridwide" { SyncScope::Grid }
            / "blockwide" { SyncScope::Block }
            / "systemwide" { SyncScope::System }

        rule float_literal() -> Expression
            = n:$(['0'..='9']+ "." ['0'..='9']* "f"?) {
                Expression::FloatLiteral(n.trim_end_matches('f').parse::<f64>().unwrap())
            }
            / n:$(['0'..='9']* "." ['0'..='9']+ "f"?) {
                Expression::FloatLiteral(n.trim_end_matches('f').parse::<f64>().unwrap())
            }
            
        rule primary() -> Expression
            = n:$(['0'..='9']+) { Expression::IntegerLiteral(n.parse().unwrap()) }
            / float_literal()
            / i:identifier() { Expression::Variable(i) }
            / "(" _ e:expression() _ ")" { e }
    }
}