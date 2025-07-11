package(default_visibility = ["//visibility:public"])

load("@rules_python//python:defs.bzl", "py_runtime", "py_runtime_pair")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

compile_pip_requirements(
    name = "requirements",
    src = "requirements.in",
    requirements_linux = "requirements_linux.txt",
    requirements_darwin = "requirements_darwin.txt",
)

# Python toolchain
py_runtime(
    name = "python3_runtime",
    interpreter = "@python3//:bin/python",
    python_version = "PY3",
)

py_runtime_pair(
    name = "py_runtime_pair",
    py2_runtime = None,
    py3_runtime = ":python3_runtime",
)

toolchain(
    name = "py_3_toolchain",
    toolchain = ":py_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)

toolchain(
    name = "nix_cc_python_toolchain",
    toolchain = "@gcc//:toolchain",
    toolchain_type = "@rules_python//python/cc:toolchain_type",
)
