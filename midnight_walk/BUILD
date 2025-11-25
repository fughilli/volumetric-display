load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "midnight_walk_server_lib",
    srcs = [
        "server.py",
    ],
    deps = [
        requirement("itsdangerous"),
        requirement("starlette"),
        requirement("fastapi"),
        requirement("uvicorn"),
        requirement("websockets"),
        requirement("python-multipart"),
    ],
)

py_binary(
    name = "midnight_walk_server",
    srcs = ["server.py"],
    main = "server.py",
    deps = [
        ":midnight_walk_server_lib",
        requirement("fastapi"),
        requirement("uvicorn"),
        requirement("websockets"),
    ],
)

py_package(
    name = "midnight_walk_server_pkg",
    deps = [":midnight_walk_server_lib"],
)

py_wheel(
    name = "midnight_walk_server_wheel",
    distribution = "midnight_walk_server",
    entry_points = {"console_scripts": ["midnight-walk = midnight_walk.server:main"]},
    python_tag = "py3",
    requires = [
        "fastapi",
        "itsdangerous",
        "python-multipart",
        "starlette",
        "uvicorn",
        "websockets",
    ],
    version = "0.0.1",
    deps = [":midnight_walk_server_pkg"],
)
