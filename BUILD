load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")
load("@pip//:requirements.bzl", "requirement")
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
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
        ":display_config",
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
        ":display_config",
        ":volumetric_display",
        "//resources:icon",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log:flags",
        "@abseil-cpp//absl/log:initialize",
        "@glm",
        "@nlohmann_json//:json",
    ],
)

pkg_tar(
    name = "simulator_pkg",
    srcs = [":simulator"],
    include_runfiles = True,
)

py_binary(
    name = "discover",
    srcs = ["discover.py"],
)

py_library(
    name = "artnet",
    srcs = ["artnet.py"],
    deps = ["//src/artnet:artnet_rs"],
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

cc_library(
    name = "display_config",
    hdrs = [":DisplayConfig.h"],
)

py_library(
    name = "control_port_rust",
    srcs = ["control_port_rust.py"],
    visibility = ["//visibility:public"],
    deps = ["//src/control_port:control_port_rs"],
)

py_library(
    name = "sender_monitor_rust",
    srcs = ["sender_monitor_rust.py"],
    visibility = ["//visibility:public"],
    deps = ["//src/sender_monitor:sender_monitor_rs"],
)

py_binary(
    name = "sender",
    srcs = ["sender.py"],
    data = [
        "//sounds:chuck_sources",
        "//sounds:rendered_sounds",  # Pre-rendered MP3 sound files
        "//sounds:sound_server_launcher",
        "@chuck//:bin/chuck",
    ],
    deps = [
        ":artnet",
        ":control_port_rust",
        ":sender_monitor_rust",
        "//games/util:sound_manager",
        "@rules_python//python/runfiles",
        requirement("numpy"),
    ],
)

pkg_tar(
    name = "sender_pkg",
    srcs = [":sender"],
    include_runfiles = True,
)

py_test(
    name = "test_sender_monitor",
    srcs = ["test_sender_monitor.py"],
    python_version = "PY3",
    deps = [":sender_monitor_rust"],
)

py_test(
    name = "test_rust_control_port",
    srcs = ["test_rust_control_port.py"],
    python_version = "PY3",
    deps = [
        ":control_port_rust",
        "//games/util:game_util_rust",
    ],
)

py_binary(
    name = "test_sound_system",
    srcs = ["test_sound_system.py"],
    data = [
        "//sounds:rendered_sounds",  # Pre-rendered MP3 sound files
    ],
    deps = [
        "//games/util:sound_manager",
        "@rules_python//python/runfiles",
    ],
)

py_binary(
    name = "test_all_sounds",
    srcs = ["test_all_sounds.py"],
    data = [
        "//sounds:rendered_sounds",  # Pre-rendered MP3 sound files
    ],
    deps = [
        "//games/util:sound_manager",
        "@rules_python//python/runfiles",
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

py_library(
    name = "controller_simulator_lib",
    srcs = ["controller_simulator_lib.py"],
    visibility = ["//visibility:public"],
)
