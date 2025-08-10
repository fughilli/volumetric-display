load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "control_port_integration_test",
    size = "large",
    timeout = "long",
    srcs = ["control_port_integration_test.py"],
    python_version = "PY3",
    deps = [
        "//:controller_simulator_lib",
    ],
)

py_test(
    name = "real_control_port_integration_test",
    size = "large",
    timeout = "long",
    srcs = ["real_control_port_integration_test.py"],
    python_version = "PY3",
    deps = [
        "//:control_port_rust",
        "//:controller_simulator_lib",
    ],
)
