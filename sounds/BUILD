load("@rules_python//python:defs.bzl", "py_binary")
load(":chuck_render.bzl", "chuck_render")

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

# Render Chuck files to WAV
chuck_render(
    name = "beep_wav",
    src = "beep.ck",
    duration = "1.0",
    output = "beep.wav",
)

chuck_render(
    name = "click_wav",
    src = "click.ck",
    duration = "0.5",
    output = "click.wav",
)

chuck_render(
    name = "crunch_wav",
    src = "crunch.ck",
    duration = "1.0",
    output = "crunch.wav",
)

chuck_render(
    name = "ding_wav",
    src = "ding.ck",
    duration = "1.0",
    output = "ding.wav",
)

chuck_render(
    name = "pop_wav",
    src = "pop.ck",
    duration = "0.5",
    output = "pop.wav",
)

chuck_render(
    name = "thud_wav",
    src = "thud.ck",
    duration = "1.0",
    output = "thud.wav",
)

chuck_render(
    name = "warble_wav",
    src = "warble.ck",
    duration = "2.0",
    output = "warble.wav",
)

chuck_render(
    name = "woosh_wav",
    src = "woosh.ck",
    duration = "1.5",
    output = "woosh.wav",
)

# Filegroup containing all rendered WAV files
filegroup(
    name = "rendered_sounds",
    srcs = [
        ":beep_wav",
        ":click_wav",
        ":crunch_wav",
        ":ding_wav",
        ":pop_wav",
        ":thud_wav",
        ":warble_wav",
        ":woosh_wav",
    ],
    visibility = ["//visibility:public"],
)

# Python sound manager library is now in //games/util:sound_manager

# Chuck sound server launcher
py_binary(
    name = "sound_server_launcher",
    srcs = ["sound_server_launcher.py"],
    data = [
        ":chuck_sources",
        "@chuck//:bin/chuck",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@rules_python//python/runfiles",
    ],
)
