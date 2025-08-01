load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

py_binary(
    name = "mapper",
    srcs = ["mapper.py"],
    deps = [
        requirement("python-rtmidi"),
        requirement("python-osc"),
    ],
)
