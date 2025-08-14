workspace(name = "volumetric-display")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# =============================================================================
# Load dependencies.
# =============================================================================

http_archive(
    name = "io_tweag_rules_nixpkgs",
    sha256 = "30271f7bd380e4e20e4d7132c324946c4fdbc31ebe0bbb6638a0f61a37e74397",
    strip_prefix = "rules_nixpkgs-0.13.0",
    urls = ["https://github.com/tweag/rules_nixpkgs/releases/download/v0.13.0/rules_nixpkgs-0.13.0.tar.gz"],
)

http_archive(
    name = "rules_sh",
    sha256 = "3243af3fcb3768633fd39f3654de773e5fb61471a2fae5762a1653c22c412d2c",
    strip_prefix = "rules_sh-0.4.0",
    urls = ["https://github.com/tweag/rules_sh/releases/download/v0.4.0/rules_sh-0.4.0.tar.gz"],
)

# nlohmann_json library
# This rule downloads the specified version of the library from GitHub.
http_archive(
    name = "nlohmann_json",
    # Since nlohmann/json is a header-only library, we create a simple BUILD file
    # for it directly within the http_archive rule.
    build_file_content = """
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "json",
    hdrs = glob(["nlohmann/**/*.hpp"]),
    includes = ["."],
    visibility = ["//visibility:public"],
)
""",
    sha256 = "34660b5e9a407195d55e8da705ed26cc6d175ce5a6b1fb957e701fb4d5b04022",
    strip_prefix = "json-3.12.0/include",
    # It's good practice to use a specific version for reproducibility.
    # You can find the latest version and its SHA256 hash on the releases page:
    # https://github.com/nlohmann/json/releases
    urls = ["https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.zip"],
)

# =============================================================================
# Configure repositories.
# =============================================================================

#
# rules_nixpkgs
#

load("@io_tweag_rules_nixpkgs//nixpkgs:repositories.bzl", "rules_nixpkgs_dependencies")

rules_nixpkgs_dependencies()

load("@io_tweag_rules_nixpkgs//nixpkgs:nixpkgs.bzl", "nixpkgs_git_repository", "nixpkgs_sh_posix_configure")

nixpkgs_git_repository(
    name = "nixpkgs",
    revision = "23.11",
)

load("//third_party:packages.bzl", "register_packages")

register_packages()

nixpkgs_sh_posix_configure(
    name = "nixpkgs_posix",
)

#
# rules_sh
#

load("@rules_sh//sh:repositories.bzl", "rules_sh_dependencies")

rules_sh_dependencies()
