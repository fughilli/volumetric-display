load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@py_deps//:requirements.bzl", "requirement")
load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

cc_library(
    name = "volumetric_display",
    srcs = ["VolumetricDisplay.cpp"],
    hdrs = ["VolumetricDisplay.h"],
    copts = [
        # gluPerspective is deprecated in macOS 10.9
        "-Wno-deprecated-declarations",
    ],
    deps = [
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
        ],
    }),
    deps = [
        ":volumetric_display",
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
    deps = [requirement("netifaces")],
)

py_library(
    name = "artnet",
    srcs = ["artnet.py"],
)

py_binary(
    name = "sender",
    srcs = ["sender.py"],
    deps = [
        ":artnet",
        requirement("numpy"),
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
