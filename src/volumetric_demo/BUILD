load("@rules_rust//rust:defs.bzl", "rust_binary")

rust_binary(
    name = "volumetric_demo",
    srcs = [
        "main.rs",
        "model_loader.rs",
    ],
    data = [
        "//resources/deer_skull:deer_skull_files",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//src/artnet:artnet_rs",
        "//src/volumetric_rendering",
        "@crates_in_workspace//:anyhow",
        "@crates_in_workspace//:glam",
        "@crates_in_workspace//:glow",
        "@crates_in_workspace//:glutin",
        "@crates_in_workspace//:glutin-winit",
        "@crates_in_workspace//:image",
        "@crates_in_workspace//:raw-window-handle",
        "@crates_in_workspace//:tobj",
        "@crates_in_workspace//:winit",
    ],
)
