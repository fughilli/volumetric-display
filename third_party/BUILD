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

# Python toolchain
py_runtime(
    name = "python3_runtime",
    interpreter = "@python3//:bin/python",
    interpreter_version_info = {
        "major": "3",
        "minor": "11",
    },
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

cc_library(
    name = "py_headers",
    hdrs = ["@python3//:include"],
)

cc_library(
    name = "py_libs",
    srcs = ["@python3//:lib"],
    linkopts = [
      "-Lexternal/python3/lib/python3.11/lib-dynload",
    ],
)

py_cc_toolchain(
    name = "nix_py_cc_toolchain",
    headers = ":py_headers",
    libs = ":py_libs",
    python_version = "3.11",
)

toolchain(
    name = "nix_cc_python_toolchain",
    toolchain = ":nix_py_cc_toolchain",
    toolchain_type = "@@rules_python~//python/cc:toolchain_type",
)

#toolchain(
#    name = "nix_cc_toolchain",
#    toolchain = "@gcc//:cc-compiler-darwin_arm64",
#    toolchain_type = "@rules_cc//cc:toolchain_type",
#)
