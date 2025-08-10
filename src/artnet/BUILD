load("@rules_pyo3//pyo3:defs.bzl", "pyo3_extension")

pyo3_extension(
    name = "artnet_rs",
    srcs = ["lib.rs"],
    crate_features = [
        "extension-module",
        "abi3-py311",
    ],
    visibility = ["//visibility:public"],
)
