load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "glew",
    srcs = glob([
        "**/*.dylib",
        "**/*.so",
    ]),
    hdrs = glob([
        "include/GL/*.h",
        "include/GL/*.hpp",
    ]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
