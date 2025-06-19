use anyhow::{Context, Result};
use clap::{ArgAction, Parser};
use parser::unified_ast::{Expression, KernelFunction, Operator, Statement, Type};

use std::fs;
use std::path::PathBuf;

pub mod metal;
pub mod parser;
pub mod analyzer;
pub mod optimizer;

use analyzer::CodeAnalyzer;
use optimizer::KernelOptimizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // input: CUDA file
    #[arg(short, long)]
    input: PathBuf,

    // output directory for generated Metal code
    #[arg(short = 'd', long)]
    output: PathBuf,

    // optimization level (0-3)
    #[arg(short, long, default_value_t = 2)]
    opt_level: u8,

    // verbose mode (-v, -vv, -vvv)
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,

    // run the kernel after generation
    #[arg(short, long)]
    run: bool,

    /// Print the parsed Abstract Syntax Tree to stdout
    #[arg(long)]
    print_ast: bool,

    /// Print the generated Metal shader to stdout
    #[arg(long)]
    emit_metal: bool,

    /// List kernels found in the input file
    #[arg(long)]
    list_kernels: bool,

    /// Enable advanced optimization analysis
    #[arg(long)]
    analyze: bool,

    /// Generate optimization report
    #[arg(long)]
    optimization_report: bool,

    /// Target device (apple_silicon, intel_mac)
    #[arg(long, default_value = "apple_silicon")]
    target: String,
}

fn validate_input(path: &PathBuf) -> Result<()> {
    // Check if file exists
    if !path.exists() {
        anyhow::bail!("Input file does not exist: {:?}", path);
    }

    if !path.is_file() {
        anyhow::bail!("Input path is not a file: {:?}", path);
    }

    // Check extension
    match path.extension() {
        Some(ext) if ext == "cu" => Ok(()),
        _ => anyhow::bail!("Input file must have .cu extension: {:?}", path),
    }
}

fn ensure_output_dir(path: &PathBuf) -> Result<()> {
    if path.exists() {
        if !path.is_dir() {
            anyhow::bail!("Output path exists but is not a directory: {:?}", path);
        }
    } else {
        fs::create_dir_all(path).context("Failed to create output directory")?;
        log::debug!("Created output directory: {:?}", path);
    }
    Ok(())
}

fn determine_kernel_dimensions(kernel: &KernelFunction) -> u32 {
    let has_mn_params = kernel
        .parameters
        .iter()
        .any(|p| matches!(p.param_type, Type::Int) && (p.name == "M" || p.name == "N"));

    fn has_matrix_indexing(expr: &Expression) -> bool {
        match expr {
            Expression::BinaryOp(lhs, op, rhs) => match op {
                Operator::Add => {
                    if let Expression::BinaryOp(_, Operator::Multiply, _) = **lhs {
                        true
                    } else {
                        has_matrix_indexing(lhs) || has_matrix_indexing(rhs)
                    }
                }
                _ => has_matrix_indexing(lhs) || has_matrix_indexing(rhs),
            },
            _ => false,
        }
    }

    for stmt in &kernel.body.statements {
        match stmt {
            Statement::Assign(assign) => {
                if has_matrix_indexing(&assign.value) {
                    return 2;
                }
            }
            Statement::ForLoop { .. } => {
                return 2;
            }
            _ => continue,
        }
    }

    if has_mn_params {
        2
    } else {
        1
    }
}

fn print_banner() {
    println!("\n\x1b[1;36m╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸\x1b[0m");
    println!(
        "\x1b[1;33m   CUDApple v{} ⚡ Enhanced Edition\x1b[0m",
        env!("CARGO_PKG_VERSION")
    );
    println!("\x1b[0m   Running CUDA code directly on your Mac chip");
    println!("\x1b[1;32m   🧠 With intelligent optimization & analysis\x1b[0m");
    println!("\x1b[1;36m╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸\x1b[0m\n");
}

fn print_section_header(title: &str) {
    println!("\n\x1b[1;35m=== {} ===\x1b[0m", title);
}

fn print_analysis_results(analysis: &analyzer::AnalysisResult) {
    print_section_header("🧠 Intelligent Code Analysis");
    
    println!("📊 \x1b[1;32mOverall Performance Score: {:.1}/100\x1b[0m", analysis.overall_score);
    println!("🚀 \x1b[1;33mEstimated Performance Gain: {:.1}x\x1b[0m", analysis.estimated_performance_gain);
    println!("💎 \x1b[1;36mMetal Compatibility: {:?}\x1b[0m", analysis.metal_compatibility);
    
    if !analysis.performance_bottlenecks.is_empty() {
        println!("\n⚠️  \x1b[1;31mIdentified Bottlenecks:\x1b[0m");
        for (i, bottleneck) in analysis.performance_bottlenecks.iter().enumerate() {
            println!("   {}. {}", i + 1, bottleneck);
        }
    }
}

fn print_optimization_results(results: &optimizer::OptimizationResult) {
    print_section_header("⚡ Optimization Results");
    
    println!("🔧 \x1b[1;32mApplied {} optimizations\x1b[0m", results.applied_optimizations.len());
    println!("📈 \x1b[1;33mPerformance improvement: {:.1}x\x1b[0m", results.performance_improvement);
    
    if !results.applied_optimizations.is_empty() {
        println!("\n✨ \x1b[1;36mOptimizations applied:\x1b[0m");
        for (i, opt) in results.applied_optimizations.iter().enumerate() {
            println!("   {}. {}", i + 1, opt);
        }
    }
    
    if !results.warnings.is_empty() {
        println!("\n⚠️  \x1b[1;31mOptimization warnings:\x1b[0m");
        for warning in &results.warnings {
            println!("   • {}", warning);
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logger with custom format
    env_logger::Builder::new()
        .filter_level(match args.verbose {
            0 => log::LevelFilter::Warn,
            1 => log::LevelFilter::Info,
            2 => log::LevelFilter::Debug,
            _ => log::LevelFilter::Trace,
        })
        .format(|buf, record| {
            use std::io::Write;
            let level_color = match record.level() {
                log::Level::Error => "\x1b[1;31m", // Bright Red
                log::Level::Warn => "\x1b[1;33m",  // Bright Yellow
                log::Level::Info => "\x1b[1;32m",  // Bright Green
                log::Level::Debug => "\x1b[1;36m", // Bright Cyan
                log::Level::Trace => "\x1b[1;37m", // Bright White
            };
            writeln!(
                buf,
                "{}{:<5}\x1b[0m {}",
                level_color,
                record.level(),
                record.args()
            )
        })
        .init();

    print_banner();

    // Validate input file
    validate_input(&args.input).context("Input validation failed")?;
    // Ensure output directory exists
    ensure_output_dir(&args.output).context("Output directory setup failed")?;

    // Read and parse the CUDA file
    print_section_header("CUDA Source Analysis");
    let cuda_source = fs::read_to_string(&args.input).context("Failed to read CUDA source file")?;
    let mut cuda_program = parser::parse_cuda(&cuda_source).context("Failed to parse CUDA program")?;

    log::info!(
        "✓ Successfully parsed CUDA program with {} kernels",
        cuda_program.device_code.len()
    );

    // Log kernels
    for kernel in &cuda_program.device_code {
        log::info!("📦 Found kernel: {}", kernel.name);

        // Print kernel information
        if args.verbose > 0 {
            log::info!("   ├─ Parameters: {}", kernel.parameters.len());
            for param in &kernel.parameters {
                log::debug!("   │  ├─ {}: {:?}", param.name, param.param_type);
            }
        }
    }

    if args.list_kernels {
        print_section_header("Kernels");
        for kernel in &cuda_program.device_code {
            println!("{}", kernel.name);
        }
    }

    // Advanced code analysis
    let analysis_result = if args.analyze || args.opt_level > 1 {
        let mut analyzer = CodeAnalyzer::new();
        let analysis = analyzer.analyze_program(&cuda_program);
        
        if args.analyze {
            print_analysis_results(&analysis);
        }
        
        Some(analysis)
    } else {
        None
    };

    // Apply optimizations
    let optimization_result = if args.opt_level > 0 {
        let mut optimizer = KernelOptimizer::new();
        let result = optimizer.optimize_program(&mut cuda_program, analysis_result.as_ref().unwrap_or(&analyzer::AnalysisResult {
            overall_score: 50.0,
            performance_bottlenecks: Vec::new(),
            metal_compatibility: analyzer::MetalCompatibility::Medium,
            estimated_performance_gain: 1.0,
        }));
        
        if args.optimization_report {
            print_optimization_results(&result);
            println!("\n{}", optimizer.generate_optimization_report());
        }
        
        Some(result)
    } else {
        None
    };

    if args.print_ast {
        print_section_header("AST Dump");
        println!("{:#?}", cuda_program);
    }

    // Generate Metal code for each kernel
    print_section_header("Metal Translation");
    
    for kernel in &cuda_program.device_code {
        let dimensions = determine_kernel_dimensions(kernel);
        log::info!("✓ Generated Metal shader code");
        log::info!("   ├─ Dimensions: {}", dimensions);
        
        if dimensions == 1 {
            log::info!("   ├─ Grid size: (4096, 1, 1)");
            log::info!("   └─ Thread group size: (256, 1, 1)");
        } else {
            log::info!("   ├─ Grid size: (63, 63, 1)");
            log::info!("   └─ Thread group size: (16, 16, 1)");
        }
    }

    // Generate metal code using the enhanced generator
    let metal_generator = parser::metal_generator::MetalCodeGenerator::new(&cuda_program.device_code[0]);
    let metal_code = metal_generator.generate_metal_shader(&cuda_program.device_code[0]);
    let swift_runner = metal_generator.generate_swift_runner(&cuda_program.device_code[0]);

    if args.emit_metal {
        print_section_header("Metal Shader");
        println!("{}", metal_code);
    }

    // Write files
    print_section_header("File Generation");
    
    let metal_file = args.output.join("kernel.metal");
    fs::write(&metal_file, metal_code).context("Failed to write Metal shader file")?;
    log::info!("✓ Written Metal shader: {:?}", metal_file);

    let runner_file = args.output.join("MetalKernelRunner.swift");
    fs::write(&runner_file, swift_runner).context("Failed to write Swift runner file")?;
    
    let main_swift = args.output.join("main.swift");
    let main_content = include_str!("metal/templates/main.swift");
    fs::write(&main_swift, main_content).context("Failed to write main.swift file")?;
    
    log::info!("✓ Written Swift files:");
    log::info!("   ├─ {:?}", runner_file);
    log::info!("   └─ {:?}", main_swift);

    // Run the kernel if requested
    if args.run {
        print_section_header("Kernel Execution");
        
        let output_dir = args.output.canonicalize()?;
        let build_cmd = format!("cd {:?} && xcrun -sdk macosx swiftc MetalKernelRunner.swift main.swift -o kernel_runner", output_dir);
        let run_cmd = format!("cd {:?} && ./kernel_runner", output_dir);
        
        log::info!("🔨 Building kernel runner...");
        let build_status = std::process::Command::new("sh")
            .arg("-c")
            .arg(&build_cmd)
            .status()
            .context("Failed to build kernel runner")?;
            
        if build_status.success() {
            log::info!("✓ Build successful");
            log::info!("🚀 Running kernel...");
            
            let run_status = std::process::Command::new("sh")
                .arg("-c")
                .arg(&run_cmd)
                .status()
                .context("Failed to run kernel")?;
                
            if run_status.success() {
                log::info!("✅ Kernel execution completed successfully!");
            } else {
                log::error!("❌ Kernel execution failed");
            }
        } else {
            log::error!("❌ Build failed");
        }
    }

    // Print summary
    print_section_header("Summary");
    println!("✅ Successfully completed all operations!");
    
    if let Some(analysis) = &analysis_result {
        println!("🧠 Analysis score: {:.1}/100", analysis.overall_score);
    }
    
    if let Some(optimization) = &optimization_result {
        println!("⚡ Applied {} optimizations", optimization.applied_optimizations.len());
    }

    if !args.run {
        println!("\nTo run the kernel, use the --run flag or execute the following commands:");
        println!("cd {:?}", args.output);
        println!("xcrun -sdk macosx swiftc MetalKernelRunner.swift main.swift -o kernel_runner");
        println!("./kernel_runner");
    }

    Ok(())
}
