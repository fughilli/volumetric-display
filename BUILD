load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")
load("@pip//:requirements.bzl", "requirement")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_pyo3//pyo3:defs.bzl", "pyo3_extension")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@rules_rust//rust:defs.bzl", "rust_binary")

cc_library(
    name = "volumetric_display",
    srcs = ["VolumetricDisplay.cpp"],
    hdrs = ["VolumetricDisplay.h"],
    copts = [
        # gluPerspective is deprecated in macOS 10.9
        "-Wno-deprecated-declarations",
    ],
    deps = [
        ":color_correction",
        "@abseil-cpp//absl/log",
        "@boost",
        "@glew",
        "@glfw",
        "@glm",
    ],
)

cc_binary(
    name = "simulator",
    srcs = [":main.cpp"],
    linkopts = select({
        "@platforms//os:linux": [
            "-lGL",
            "-lGLU",
        ],
        "@platforms//os:osx": [
            "-framework",
            "OpenGL",
            "-framework",
            "Cocoa",
        ],
    }),
    deps = [
        ":volumetric_display",
        "//resources:icon",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log:flags",
        "@abseil-cpp//absl/log:initialize",
        "@glm",
    ],
)

py_binary(
    name = "discover",
    srcs = ["discover.py"],
)

py_library(
    name = "artnet",
    srcs = ["artnet.py"],
    deps = [":artnet_rs"],
)

py_binary(
    name = "sender",
    srcs = ["sender.py"],
    deps = [
        ":artnet",
        requirement("numpy"),
    ],
)

py_binary(
    name = "gen_routing_table",
    srcs = ["gen_routing_table.py"],
)

py_binary(
    name = "controller_simulator",
    srcs = ["controller_simulator.py"],
    deps = [
        requirement("pygame"),
    ],
)

refresh_compile_commands(
    name = "refresh_compile_commands",

    # Specify the targets of interest.
    # For example, specify a dict of targets and any flags required to build.
    targets = [
        ":simulator",
    ],
)

cc_library(
    name = "color_correction",
    hdrs = [":color_correction.h"],
)

py_library(
    name = "control_port",
    srcs = ["control_port.py"],
)

py_library(
    name = "control_port_rust",
    srcs = ["control_port_rust.py"],
    visibility = ["//visibility:public"],
    deps = [":artnet_rs"],
)

py_test(
    name = "control_port_test",
    srcs = ["control_port_test.py"],
    deps = [
        ":control_port",
    ],
)

py_binary(
    name = "test_rust_control_port",
    srcs = ["test_rust_control_port.py"],
    deps = [
        ":control_port_rust",
        "//games/util:game_util_rust",
    ],
)

rust_binary(
    name = "artnet_mapper",
    srcs = ["src/main.rs"],
    deps = [
        "@crates_in_workspace//:clap",
        "@crates_in_workspace//:lazy_static",
        "@crates_in_workspace//:midir",
        "@crates_in_workspace//:rosc",
        "@crates_in_workspace//:tokio",
        "@crates_in_workspace//:tracing",
        "@crates_in_workspace//:tracing-subscriber",
    ],
)

pyo3_extension(
    name = "artnet_rs",
    srcs = [
        "src/control_port.rs",
        "src/lib.rs",
        "src/web_monitor.rs",
    ],
    crate_features = [
        "extension-module",
        "abi3-py311",
    ],
    data = ["static/dashboard.html"],
    deps = [
        "@crates_in_workspace//:anyhow",
        "@crates_in_workspace//:axum",
        "@crates_in_workspace//:base64",
        "@crates_in_workspace//:bytes",
        "@crates_in_workspace//:chrono",
        "@crates_in_workspace//:dashmap",
        # "@crates_in_workspace//:pythonize",  # Removed due to pyo3 version incompatibility
        "@crates_in_workspace//:serde",
        "@crates_in_workspace//:serde_json",
        "@crates_in_workspace//:tokio",
        "@crates_in_workspace//:tower",
        "@crates_in_workspace//:tower-http",
    ],
)
