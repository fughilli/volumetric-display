load("@rules_rust//rust:defs.bzl", "rust_library")

rust_library(
    name = "volumetric_rendering",
    srcs = [
        "lib.rs",
        "renderer.rs",
        "shaders.rs",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@crates_in_workspace//:anyhow",
        "@crates_in_workspace//:glam",
        "@crates_in_workspace//:glow",
        "@crates_in_workspace//:thiserror",
    ],
)
