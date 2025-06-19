use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn prints_host_when_flag_set() {
    let mut cmd = Command::cargo_bin("cudapple").unwrap();
    cmd.current_dir("src")
        .args(["-i", "examples/vector_add.cu", "-d", "../tmp_output", "--emit-host"]);
    cmd.assert()
        .stdout(predicate::str::contains("import Metal"));
}
