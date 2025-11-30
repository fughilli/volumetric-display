load("@rules_pyo3//pyo3:defs.bzl", "pyo3_extension")
load("@rules_rust//rust:defs.bzl", "rust_library")

rust_library(
    name = "artnet_rs",
    srcs = [
        "core.rs",
        "lib.rs",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@crates_in_workspace//:pyo3",
    ],
)

pyo3_extension(
    name = "artnet_rs_py",
    srcs = ["lib.rs"],
    crate_features = [
        "extension-module",
        "abi3-py311",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":artnet_rs",
    ],
)
