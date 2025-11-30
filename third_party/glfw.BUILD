load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "glfw",
    srcs = glob([
        "**/*.dylib",
        "**/*.so",
        "**/*.a",
    ]),
    hdrs = glob([
        "include/GLFW/*.h",
        "include/GLFW/*.hpp",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "lib",
    srcs = glob([
        "**/*.dylib",
        "**/*.so",
        "**/*.a",
    ]),
    visibility = ["//visibility:public"],
)
