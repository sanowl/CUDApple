use std::fmt;

pub use self::{
    operators::Operator,
    dimensions::Dimension,
    types::Type,
};

mod operators {
    #[derive(Debug, Clone, PartialEq)]
    pub enum Operator {
        Add,
        Subtract,
        Multiply,
        Divide,
        Modulo,
        LessThan,
        LessThanEqual,
        GreaterThan,
        GreaterThanEqual,
        Equal,
        NotEqual,
        LogicalAnd,
        LogicalOr,
        LogicalNot,
        BitwiseAnd,
        BitwiseOr,
        BitwiseXor,
        BitwiseNot,
        LeftShift,
        RightShift,
        // Compound assignment operators
        AddAssign,
        SubAssign,
        MulAssign,
        DivAssign,
        ModAssign,
        AndAssign,
        OrAssign,
        XorAssign,
        LeftShiftAssign,
        RightShiftAssign,
    }
}

mod dimensions {
    #[derive(Debug, Clone, PartialEq)]
    pub enum Dimension {
        X,
        Y,
        Z,
    }
}

pub mod types {
    use super::{StructField, Qualifier};
    
    #[derive(Debug, Clone, PartialEq)]
    pub enum Type {
        Int,
        UInt,
        Long,
        ULong,
        Short,
        UShort,
        Char,
        UChar,
        Float,
        Double,
        Half,
        Bool,
        Void,
        Pointer(Box<Type>),
        Vector(Box<Type>, usize),
        Matrix(Box<Type>, usize, usize), // type, rows, cols
        Struct(String, Vec<StructField>),
        Template(String, Vec<Type>),
        Array(Box<Type>, Option<usize>),
        Texture1D(Box<Type>),
        Texture2D(Box<Type>),
        Texture3D(Box<Type>),
        Surface1D(Box<Type>),
        Surface2D(Box<Type>),
        Surface3D(Box<Type>),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: String,
    pub field_type: Type,
    pub qualifiers: Vec<Qualifier>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CudaProgram {
    pub device_code: Vec<KernelFunction>,
    pub host_code: Vec<HostFunction>,
    pub type_definitions: Vec<TypeDefinition>,
    pub constants: Vec<Constant>,
    pub textures: Vec<TextureDeclaration>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HostFunction {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub return_type: Type,
    pub body: Option<Block>,
    pub is_inline: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeDefinition {
    pub name: String,
    pub definition: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Constant {
    pub name: String,
    pub value: Expression,
    pub const_type: Type,
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
        else_body: Option<Block>,
    },
    WhileLoop {
        condition: Expression,
        body: Block,
    },
    ForLoop {
        init: Box<Statement>,
        condition: Expression,
        increment: Box<Statement>,
        body: Block,
    },
    DoWhileLoop {
        body: Block,
        condition: Expression,
    },
    Switch {
        expression: Expression,
        cases: Vec<SwitchCase>,
    },
    CompoundAssign {
        target: Expression,
        operator: Operator,
        value: Expression,
    },
    Expression(Box<Expression>),
    SyncThreads,
    SyncWarp,
    AtomicOperation {
        operation: AtomicOp,
        target: Expression,
        value: Expression,
        compare: Option<Expression>, // For CAS operations
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
        qualifiers: Vec<FunctionQualifier>,
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
    Return(Option<Expression>),
    Break,
    Continue,
    Label(String),
    Goto(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SwitchCase {
    pub value: Option<Expression>, // None for default case
    pub statements: Vec<Statement>,
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
    UIntegerLiteral(u64),
    FloatLiteral(f64),
    DoubleLiteral(f64),
    BoolLiteral(bool),
    StringLiteral(String),
    CharLiteral(char),
    Infinity,
    NegativeInfinity,
    NaN,
    UnaryOp(Operator, Box<Expression>),
    BinaryOp(Box<Expression>, Operator, Box<Expression>),
    TernaryOp(Box<Expression>, Box<Expression>, Box<Expression>), // condition ? true_val : false_val
    Cast(Type, Box<Expression>),
    ThreadIdx(Dimension),
    BlockIdx(Dimension),
    BlockDim(Dimension),
    GridDim(Dimension),
    WarpSize,
    LaneId,
    ArrayAccess {
        array: Box<Expression>,
        index: Box<Expression>,
    },
    StructAccess {
        object: Box<Expression>,
        field: String,
    },
    PointerAccess {
        object: Box<Expression>,
        field: String,
    },
    FunctionCall {
        name: String,
        arguments: Vec<Expression>,
    },
    MathFunction {
        name: String,
        arguments: Vec<Expression>,
    },
    AtomicFunction {
        name: String,
        arguments: Vec<Expression>,
    },
    TextureFunction {
        name: String,
        texture: String,
        coordinates: Vec<Expression>,
        arguments: Vec<Expression>,
    },
    SizeOf(Type),
    AlignOf(Type),
    AddressOf(Box<Expression>),
    Dereference(Box<Expression>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemorySpace {
    Global,
    Shared,
    Constant,
    Local,
    Texture,
    Surface,
    Unified,
    Restricted,
    Default,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Qualifier {
    Restrict,
    Const,
    Volatile,
    Inline,
    Static,
    Extern,
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionQualifier {
    Device,
    Host,
    Global,
    Inline,
    NoInline,
    ForceInline,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelFunction {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub body: Block,
    pub launch_bounds: Option<LaunchBounds>,
    pub shared_memory_size: Option<usize>,
    pub qualifiers: Vec<FunctionQualifier>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LaunchBounds {
    pub max_threads_per_block: usize,
    pub min_blocks_per_multiprocessor: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub param_type: Type,
    pub qualifier: Qualifier,
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
    CAS, // Compare and swap
    Inc,
    Dec,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MathFunction {
    // Trigonometric
    Sin, Cos, Tan, Asin, Acos, Atan, Atan2,
    Sinh, Cosh, Tanh, Asinh, Acosh, Atanh,
    Sincos, Sincospi,
    
    // Exponential and Logarithmic
    Exp, Exp2, Exp10, Expm1,
    Log, Log2, Log10, Log1p,
    Pow, Sqrt, Rsqrt, Cbrt,
    
    // Rounding and Absolute
    Abs, Fabs, Ceil, Floor, Round, Trunc,
    Rint, Nearbyint, Lrint, Llrint,
    
    // Min/Max and Comparison
    Max, Min, Fmax, Fmin,
    Fdim, Fmod, Remainder,
    
    // Floating-point manipulation
    Frexp, Ldexp, Modf,
    Copysign, Nextafter,
    
    // Error and Gamma functions
    Erf, Erfc, Tgamma, Lgamma,
    
    // Bessel functions
    J0, J1, Jn, Y0, Y1, Yn,
    
    // Hyperbolic and inverse hyperbolic
    Hypot, Rhypot,
    
    // Fast math functions
    FastSin, FastCos, FastTan,
    FastExp, FastLog, FastPow,
    FastSqrt, FastRsqrt,
    
    // Integer functions
    Abs32, Min32, Max32,
    Clz, Popc, Ffs, Brev,
    
    // Special constants
    Inf, NaN,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TextureType {
    Tex1D,
    Tex1DLayered,
    Tex2D,
    Tex2DLayered,
    Tex3D,
    TexCubemap,
    TexCubemapLayered,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextureDeclaration {
    pub name: String,
    pub tex_type: TextureType,
    pub element_type: Type,
    pub read_mode: TextureReadMode,
    pub address_mode: Vec<TextureAddressMode>,
    pub filter_mode: TextureFilterMode,
    pub normalized_coords: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TextureReadMode {
    ElementType,
    NormalizedFloat,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TextureAddressMode {
    Wrap,
    Clamp,
    Mirror,
    Border,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TextureFilterMode {
    Point,
    Linear,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TextureOp {
    Read,
    Write,
    Sample,
    Gather,
    Load,
    Store,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryFence {
    System,
    Global,
    Shared,
    Thread,
    Device,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SyncScope {
    Thread,
    Warp,
    Block,
    Grid,
    System,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::Int => write!(f, "int"),
            Type::UInt => write!(f, "unsigned int"),
            Type::Long => write!(f, "long"),
            Type::ULong => write!(f, "unsigned long"),
            Type::Short => write!(f, "short"),
            Type::UShort => write!(f, "unsigned short"),
            Type::Char => write!(f, "char"),
            Type::UChar => write!(f, "unsigned char"),
            Type::Float => write!(f, "float"),
            Type::Double => write!(f, "double"),
            Type::Half => write!(f, "half"),
            Type::Bool => write!(f, "bool"),
            Type::Pointer(inner) => write!(f, "{}*", inner),
            Type::Vector(base_type, size) => write!(f, "{}{}", base_type, size),
            Type::Matrix(base_type, rows, cols) => write!(f, "{}{}x{}", base_type, rows, cols),
            Type::Struct(name, _) => write!(f, "struct {}", name),
            Type::Template(name, params) => {
                write!(f, "{}<", name)?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param)?;
                }
                write!(f, ">")
            },
            Type::Array(elem_type, size) => match size {
                Some(n) => write!(f, "{}[{}]", elem_type, n),
                None => write!(f, "{}[]", elem_type),
            },
            Type::Texture1D(elem_type) => write!(f, "texture1D<{}>", elem_type),
            Type::Texture2D(elem_type) => write!(f, "texture2D<{}>", elem_type),
            Type::Texture3D(elem_type) => write!(f, "texture3D<{}>", elem_type),
            Type::Surface1D(elem_type) => write!(f, "surface1D<{}>", elem_type),
            Type::Surface2D(elem_type) => write!(f, "surface2D<{}>", elem_type),
            Type::Surface3D(elem_type) => write!(f, "surface3D<{}>", elem_type),
        }
    }
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Add => write!(f, "+"),
            Operator::Subtract => write!(f, "-"),
            Operator::Multiply => write!(f, "*"),
            Operator::Divide => write!(f, "/"),
            Operator::Modulo => write!(f, "%"),
            Operator::LessThan => write!(f, "<"),
            Operator::LessThanEqual => write!(f, "<="),
            Operator::GreaterThan => write!(f, ">"),
            Operator::GreaterThanEqual => write!(f, ">="),
            Operator::Equal => write!(f, "=="),
            Operator::NotEqual => write!(f, "!="),
            Operator::LogicalAnd => write!(f, "&&"),
            Operator::LogicalOr => write!(f, "||"),
            Operator::LogicalNot => write!(f, "!"),
            Operator::BitwiseAnd => write!(f, "&"),
            Operator::BitwiseOr => write!(f, "|"),
            Operator::BitwiseXor => write!(f, "^"),
            Operator::BitwiseNot => write!(f, "~"),
            Operator::LeftShift => write!(f, "<<"),
            Operator::RightShift => write!(f, ">>"),
            Operator::AddAssign => write!(f, "+="),
            Operator::SubAssign => write!(f, "-="),
            Operator::MulAssign => write!(f, "*="),
            Operator::DivAssign => write!(f, "/="),
            Operator::ModAssign => write!(f, "%="),
            Operator::AndAssign => write!(f, "&="),
            Operator::OrAssign => write!(f, "|="),
            Operator::XorAssign => write!(f, "^="),
            Operator::LeftShiftAssign => write!(f, "<<="),
            Operator::RightShiftAssign => write!(f, ">>="),
        }
    }
}

impl fmt::Display for MemorySpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemorySpace::Global => write!(f, "__global__"),
            MemorySpace::Shared => write!(f, "__shared__"),
            MemorySpace::Constant => write!(f, "__constant__"),
            MemorySpace::Local => write!(f, "__local__"),
            MemorySpace::Texture => write!(f, "__texture__"),
            MemorySpace::Surface => write!(f, "__surface__"),
            MemorySpace::Unified => write!(f, "__managed__"),
            MemorySpace::Restricted => write!(f, "__restrict__"),
            MemorySpace::Default => write!(f, ""),
        }
    }
}

impl fmt::Display for Qualifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Qualifier::Restrict => write!(f, "__restrict__"),
            Qualifier::Const => write!(f, "const"),
            Qualifier::Volatile => write!(f, "volatile"),
            Qualifier::Inline => write!(f, "__inline__"),
            Qualifier::Static => write!(f, "static"),
            Qualifier::Extern => write!(f, "extern"),
            Qualifier::None => write!(f, ""),
        }
    }
}

impl Default for CudaProgram {
    fn default() -> Self {
        Self {
            device_code: Vec::new(),
            host_code: Vec::new(),
            type_definitions: Vec::new(),
            constants: Vec::new(),
            textures: Vec::new(),
        }
    }
}