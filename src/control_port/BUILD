load("@rules_pyo3//pyo3:defs.bzl", "pyo3_extension")
load("@rules_rust//rust:defs.bzl", "rust_test")

pyo3_extension(
    name = "control_port_rs",
    srcs = [
        "control_port.rs",
        "lib.rs",
        "web_monitor.rs",
    ],
    crate_features = [
        "extension-module",
        "abi3-py311",
    ],
    data = ["//static:dashboard.html"],
    visibility = ["//visibility:public"],
    deps = [
        "@crates_in_workspace//:anyhow",
        "@crates_in_workspace//:axum",
        "@crates_in_workspace//:base64",
        "@crates_in_workspace//:bytes",
        "@crates_in_workspace//:chrono",
        "@crates_in_workspace//:dashmap",
        "@crates_in_workspace//:serde",
        "@crates_in_workspace//:serde_json",
        "@crates_in_workspace//:tokio",
        "@crates_in_workspace//:tower",
        "@crates_in_workspace//:tower-http",
    ],
)

rust_test(
    name = "control_port_test",
    srcs = [
        "control_port.rs",
        "lib.rs",
        "web_monitor.rs",
    ],
    crate_root = "lib.rs",
    deps = [
        "@crates_in_workspace//:anyhow",
        "@crates_in_workspace//:axum",
        "@crates_in_workspace//:base64",
        "@crates_in_workspace//:bytes",
        "@crates_in_workspace//:chrono",
        "@crates_in_workspace//:dashmap",
        "@crates_in_workspace//:pyo3",
        "@crates_in_workspace//:serde",
        "@crates_in_workspace//:serde_json",
        "@crates_in_workspace//:tokio",
        "@crates_in_workspace//:tower",
        "@crates_in_workspace//:tower-http",
    ],
)
