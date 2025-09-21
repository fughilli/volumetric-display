load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")

# Chuck source files for sound effects
filegroup(
    name = "chuck_sources",
    srcs = [
        "beep.ck",
        "click.ck",
        "crunch.ck",
        "ding.ck",
        "pop.ck",
        "sound_server.ck",
        "thud.ck",
        "warble.ck",
        "woosh.ck",
    ],
    visibility = ["//visibility:public"],
)

# Python sound manager library
py_library(
    name = "sound_manager",
    srcs = ["sound_manager.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("python-osc"),
    ],
)

# Chuck sound server launcher
py_binary(
    name = "sound_server_launcher",
    srcs = ["sound_server_launcher.py"],
    data = [":chuck_sources"],
    visibility = ["//visibility:public"],
)
