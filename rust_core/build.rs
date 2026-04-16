/// build.rs — Aethelix flight core build script.
///
/// Responsibilities:
/// 1. Declare `#[cfg(kani)]` as a known cfg so rustc doesn't warn about
///    "unexpected cfg condition name: kani" when building outside of
///    `cargo kani` (which sets it automatically).
/// 2. Trigger a re-build whenever the compiled causal graph binary changes.
fn main() {
    // Tell rustc that `#[cfg(kani)]` is an expected condition name.
    // When running under `cargo kani`, the kani toolchain sets this automatically.
    // When building normally, this prevents the "unexpected_cfgs" lint warning.
    println!("cargo::rustc-check-cfg=cfg(kani)");

    // Re-run this build script (and recompile causal_ranker.rs) whenever the
    // compiled graph binary is updated by graph_compiler.py.
    println!("cargo::rerun-if-changed=../causal_graph/causal_graph.bin");
    println!("cargo::rerun-if-changed=../causal_graph/graph_definition.py");
    println!("cargo::rerun-if-changed=build.rs");
}
