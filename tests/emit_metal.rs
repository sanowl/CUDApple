use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn prints_metal_when_flag_set() {
    let mut cmd = Command::cargo_bin("cudapple").unwrap();
    cmd.current_dir("src")
        .args(["-i", "examples/vector_add.cu", "-d", "../tmp_output", "--emit-metal"]);
    cmd.assert()
        .stdout(predicate::str::contains("#include <metal_stdlib>"));
}
