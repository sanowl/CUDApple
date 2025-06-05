use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct CudaProgram {
    pub device_code: Vec<KernelFunction>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    VariableDecl(Declaration),
    Assign(Assignment),
    IfStmt {
        condition: Expression,
        body: Block,
    },
    ForLoop {
        init: Box<Statement>,
        condition: Expression,
        increment: Box<Statement>,
        body: Block,
    },
    CompoundAssign {
        target: Expression,
        operator: Operator,
        value: Expression,
    },
    Expression(Box<Expression>),
    SyncThreads,  // __syncthreads()
    AtomicOperation {
        operation: AtomicOp,
        target: Expression,
        value: Expression,
    },
    StructDecl {
        name: String,
        fields: Vec<Declaration>,
    },
    DeviceFunction {
        name: String,
        parameters: Vec<Parameter>,
        return_type: Type,
        body: Block,
    },
    TextureDecl(TextureDeclaration),
    TextureOperation {
        operation: TextureOp,
        texture: String,
        coordinates: Vec<Expression>,
        value: Option<Expression>,
    },
    MemoryFence(MemoryFence),
    MemoryBarrier {
        scope: SyncScope,
        fence: MemoryFence,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    pub var_type: Type,
    pub name: String,
    pub initializer: Option<Expression>,
    pub memory_space: MemorySpace,
    pub qualifiers: Vec<Qualifier>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub target: Expression,
    pub value: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Variable(String),
    IntegerLiteral(i64),
    FloatLiteral(f32),
    Infinity,
    NegativeInfinity,
    BinaryOp(Box<Expression>, Operator, Box<Expression>),
    ThreadIdx(Dimension),
    BlockIdx(Dimension),
    BlockDim(Dimension),
    ArrayAccess {
        array: Box<Expression>,
        index: Box<Expression>,
    },
    MathFunction {
        name: String,
        arguments: Vec<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Float,
    Void,
    Pointer(Box<Type>),
    Vector(Box<Type>, usize),  // For float4, int2 etc
    Struct(String),  // For user-defined structs
    Template(String, Vec<Type>),  // For template types
    Array(Box<Type>, Option<usize>),  // For fixed and dynamic arrays
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemorySpace {
    Global,
    Shared,
    Constant,
    Texture,
    Default,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::Int => write!(f, "int"),
            Type::Float => write!(f, "float"),
            Type::Pointer(inner) => write!(f, "{}*", inner),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dimension {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    Add,
    Subtract,
    Multiply,
    Divide,
    LessThan,
    LogicalAnd,
    LogicalOr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelFunction {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub param_type: Type,
    pub qualifier: Qualifier,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Qualifier {
    Restrict,
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AtomicOp {
    Add,
    Sub,
    Exchange,
    Min,
    Max,
    And,
    Or,
    Xor,
    CAS,  // Compare and swap
}

#[derive(Debug, Clone, PartialEq)]
pub enum MathFunction {
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Sqrt,
    Pow,
    Max,
    Min,
    Abs,
    Floor,
    Ceil,
    Round,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TextureType {
    Tex1D,
    Tex2D,
    Tex3D,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextureDeclaration {
    pub name: String,
    pub tex_type: TextureType,
    pub element_type: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TextureOp {
    Read,
    Write,
    Sample,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryFence {
    System,
    Global,
    Shared,
    Thread,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SyncScope {
    Block,
    Grid,
    System,
}
