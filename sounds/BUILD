load("@rules_python//python:defs.bzl", "py_binary")

# Chuck source files for sound effects
filegroup(
    name = "chuck_sources",
    srcs = [
        "beep.ck",
        "click.ck",
        "crunch.ck",
        "ding.ck",
        "pop.ck",
        "sound_server.ck",
        "thud.ck",
        "warble.ck",
        "woosh.ck",
    ],
    visibility = ["//visibility:public"],
)

# Python sound manager library is now in //games/util:sound_manager

# Chuck sound server launcher
py_binary(
    name = "sound_server_launcher",
    srcs = ["sound_server_launcher.py"],
    data = [":chuck_sources"],
    visibility = ["//visibility:public"],
)
