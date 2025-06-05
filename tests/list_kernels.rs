use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn lists_kernels_when_flag_set() {
    let mut cmd = Command::cargo_bin("cudapple").unwrap();
    cmd.current_dir("src")
        .args(["-i", "examples/vector_add.cu", "-d", "../tmp_output", "--list-kernels"]);
    cmd.assert()
        .stdout(predicate::str::contains("vectorAdd"));
}
