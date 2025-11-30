load("@rules_rust//rust:defs.bzl", "rust_binary")

rust_binary(
    name = "volumetric_demo",
    srcs = ["main.rs"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/artnet:artnet_rs",
        "//src/volumetric_rendering",
        "@crates_in_workspace//:anyhow",
        "@crates_in_workspace//:glam",
        "@crates_in_workspace//:glfw",
        "@crates_in_workspace//:glow",
        "@glfw",
    ],
    #data = [
    #    "@glfw//:lib",
    #],

    #rustc_flags = [
    #    "-C", "link-arg=-L$(locations @glfw//:lib)/lib",
    #]
)
