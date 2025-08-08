load("@py_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "mapper",
    srcs = ["mapper.py"],
    deps = [
        requirement("python-rtmidi"),
        requirement("python-osc"),
    ],
)
