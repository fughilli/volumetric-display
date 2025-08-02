package(default_visibility = ["//visibility:public"])

load("@rules_multirun//:defs.bzl", "multirun")
load("@rules_uv//uv:pip.bzl", "pip_compile")

pip_compile(
    name = "generate_requirements_linux_txt",
    python_platform = "x86_64-unknown-linux-gnu",
    requirements_in = ":requirements.in",
    requirements_txt = "requirements_linux.txt",
)

pip_compile(
    name = "generate_requirements_darwin_txt",
    python_platform = "aarch64-apple-darwin",
    requirements_in = ":requirements.in",
    requirements_txt = "requirements_darwin.txt",
)

multirun(
    name = "generate_requirements_lock",
    commands = [
        ":generate_requirements_linux_txt",
        ":generate_requirements_darwin_txt",
    ],
    # Running in a single threaded mode allows consecutive `uv` invocations to benefit
    # from the `uv` cache from the first run.
    jobs = 1,
)
