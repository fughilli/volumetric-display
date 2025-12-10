load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "sound_manager",
    srcs = [":sound_manager.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("pygame"),
    ],
)

py_library(
    name = "base_game",
    srcs = [":base_game.py"],
    deps = [
        ":sound_manager",
    ],
)

py_library(
    name = "game_util_rust",
    srcs = [
        ":base_game",
        ":game_util_rust.py",
        "//:control_port_rust",
    ],
    visibility = ["//visibility:public"],
)
