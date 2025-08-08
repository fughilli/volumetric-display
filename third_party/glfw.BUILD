load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "glfw",
    srcs = glob([
        "**/*.dylib",
        "**/*.so",
    ]),
    hdrs = glob([
        "include/GLFW/*.h",
        "include/GLFW/*.hpp",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
