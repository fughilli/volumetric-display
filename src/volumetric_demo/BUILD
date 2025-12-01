load("@rules_rust//rust:defs.bzl", "rust_binary")

rust_binary(
    name = "volumetric_demo",
    srcs = [
        "main.rs",
        "model_loader.rs",
    ],
    data = [
        "//resources/deer_skull:deer_skull_files",
        "@glfw//:lib",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//3rdparty/crates:glfw",
        "//src/artnet:artnet_rs",
        "//src/volumetric_rendering",
        "@crates_in_workspace//:anyhow",
        "@crates_in_workspace//:glam",
        "@crates_in_workspace//:glow",
        "@crates_in_workspace//:image",
        "@crates_in_workspace//:tobj",
    ],
    #data = [
    #    "@glfw//:lib",
    #],

    #rustc_flags = [
    #    "-C", "link-arg=-L$(locations @glfw//:lib)/lib",
    #]
)
