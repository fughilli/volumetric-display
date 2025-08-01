workspace(name = "volumetric-display")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# =============================================================================
# Load dependencies.
# =============================================================================

http_archive(
    name = "io_tweag_rules_nixpkgs",
    sha256 = "1adb04dc0416915fef427757f4272c4f7dacefeceeefc50f683aec7f7e9b787a",
    strip_prefix = "rules_nixpkgs-0.12.0",
    urls = ["https://github.com/tweag/rules_nixpkgs/releases/download/v0.12.0/rules_nixpkgs-0.12.0.tar.gz"],
)

http_archive(
    name = "rules_sh",
    sha256 = "3243af3fcb3768633fd39f3654de773e5fb61471a2fae5762a1653c22c412d2c",
    strip_prefix = "rules_sh-0.4.0",
    urls = ["https://github.com/tweag/rules_sh/releases/download/v0.4.0/rules_sh-0.4.0.tar.gz"],
)

# =============================================================================
# Configure repositories.
# =============================================================================

#
# rules_nixpkgs
#

load("@io_tweag_rules_nixpkgs//nixpkgs:repositories.bzl", "rules_nixpkgs_dependencies")

rules_nixpkgs_dependencies()

load("@io_tweag_rules_nixpkgs//nixpkgs:nixpkgs.bzl", "nixpkgs_cc_configure", "nixpkgs_git_repository", "nixpkgs_package", "nixpkgs_sh_posix_configure")

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

nixpkgs_cc_configure(
  name = "gcc",
  repository = "@nixpkgs",
  nix_file_content = "(import <nixpkgs> {}).gcc11",
  cc_std = "gnu++20",
  register = True,
)

register_toolchains("//third_party:nix_cc_python_toolchain")
