use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    emit_build_version();

    tonic_build::configure()
        .build_server(false)
        .build_client(true)
        .compile_protos(&["../../proto/data.proto"], &["../../proto"])?;

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["../../proto/gossip.proto"], &["../../proto"])?;

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["../../proto/compute.proto"], &["../../proto"])?;

    Ok(())
}

/// Make the version the binary reports follow the Git tag automatically, so a
/// release tagged `v0.2.5` reports `0.2.5` without anyone editing Cargo.toml
/// (which would only ever be stale by the next tag). Exposed to the crate as the
/// `DIFFUSE_VERSION` compile-time env var.
///
/// Resolution order:
///   1. A `DIFFUSE_VERSION` already set in the build environment — lets a CI job
///      or a distro packager pin the string explicitly (e.g. building from a
///      tarball with no `.git`).
///   2. `git describe --tags --always --dirty`, with a leading `v` stripped so
///      it reads as a plain semver. On an exact tag this is `0.2.5`; a few
///      commits past it becomes `0.2.5-3-gabc1234`, and a dirty tree gains a
///      `-dirty` suffix.
///   3. The workspace `CARGO_PKG_VERSION` as a last-resort fallback.
fn emit_build_version() {
    let version = std::env::var("DIFFUSE_VERSION")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .or_else(git_describe)
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());

    println!("cargo:rustc-env=DIFFUSE_VERSION={}", version);

    // Recompute when HEAD moves or a tag/ref changes, so the embedded version
    // stays in step with the checkout without forcing a rebuild every time.
    for path in ["../../.git/HEAD", "../../.git/packed-refs"] {
        if std::path::Path::new(path).exists() {
            println!("cargo:rerun-if-changed={}", path);
        }
    }
    println!("cargo:rerun-if-env-changed=DIFFUSE_VERSION");
}

fn git_describe() -> Option<String> {
    let output = Command::new("git")
        .args(["describe", "--tags", "--always", "--dirty"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let described = String::from_utf8(output.stdout).ok()?;
    let described = described.trim();
    if described.is_empty() {
        return None;
    }
    Some(described.strip_prefix('v').unwrap_or(described).to_string())
}
