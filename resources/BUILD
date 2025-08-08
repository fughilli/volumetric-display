load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "icon",
    srcs = select({
        "@platforms//os:osx": ["icon_helper_osx.cpp"],
        "@platforms//os:linux": ["icon_helper_linux.cpp"],
    }) + ["icon.cpp"],
    hdrs = [
        "icon.h",
        "icon_helper.h",
    ],
    copts = select({
        "@platforms//os:osx": [
            "-x",
            "objective-c++",
            "-fobjc-arc",
        ],
        "//conditions:default": [],
    }),
    data = [
        "icon.png",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/log",
        "@bazel_tools//tools/cpp/runfiles",
    ],
)
