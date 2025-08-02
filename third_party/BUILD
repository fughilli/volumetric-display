package(default_visibility = ["//visibility:public"])

load("@rules_python//python:defs.bzl", "py_runtime", "py_runtime_pair")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@rules_python//python/cc:py_cc_toolchain.bzl", "py_cc_toolchain")

compile_pip_requirements(
    name = "requirements",
    src = "requirements.in",
    requirements_darwin = "requirements_darwin.txt",
    requirements_linux = "requirements_linux.txt",
)
