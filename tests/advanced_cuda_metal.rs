use cudapple::parser;
use cudapple::parser::metal_generator::MetalCodeGenerator;

#[test]
fn emits_real_cuda_launch_semantics_for_metal() {
    let cuda_source = include_str!("../src/examples/vector_add.cu");
    let program = parser::parse_cuda(cuda_source).expect("vector_add should parse");
    let kernel = &program.device_code[0];
    let generator = MetalCodeGenerator::new(kernel);

    let metal = generator.generate_metal_shader(kernel);

    assert!(metal.contains("cuda_thread_idx [[thread_position_in_threadgroup]]"));
    assert!(metal.contains("cuda_block_idx [[threadgroup_position_in_grid]]"));
    assert!(metal.contains("cuda_block_dim [[threads_per_threadgroup]]"));
    assert!(metal.contains("int(cuda_block_idx.x)"));
    assert!(metal.contains("int(cuda_block_dim.x)"));
    assert!(metal.contains("int(cuda_thread_idx.x)"));
    assert!(!metal.contains("uint i = index"));
    assert!(!metal.contains("Expression::BlockIdx"));
}

#[test]
fn host_runner_uses_explicit_output_binding_and_signed_scalars() {
    let cuda_source = include_str!("../src/examples/vector_add.cu");
    let program = parser::parse_cuda(cuda_source).expect("vector_add should parse");
    let kernel = &program.device_code[0];
    let generator = MetalCodeGenerator::new(kernel);

    let swift = generator.generate_swift_runner(kernel);

    assert!(swift.contains("outputBufferIndex: 2"));
    assert!(swift.contains("let dim = Int32(problemSize)"));
    assert!(swift.contains("(data: dim, type: Int32.self)"));
    assert!(swift.contains("dispatchThreadgroups"));
}
