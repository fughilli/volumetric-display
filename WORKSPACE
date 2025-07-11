workspace(name = "volumetric-display")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# =============================================================================
# Load dependencies.
# =============================================================================

http_archive(
    name = "rules_python",
    sha256 = "c6fb25d0ba0246f6d5bd820dd0b2e66b339ccc510242fd4956b9a639b548d113",
    strip_prefix = "rules_python-0.37.2",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.37.2/rules_python-0.37.2.tar.gz",
)

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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_pyo3",
    integrity = "sha256-8oBQFpB/6y774Y2zZij4srGRqRVUWbmWrMV/8N2ttxk=",
    urls = ["https://github.com/abrisco/rules_pyo3/releases/download/0.0.9/rules_pyo3-v0.0.9.tar.gz"],
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
# rules_python
#

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "py_deps",
    extra_pip_args = ["--no-cache-dir"],
    python_interpreter_target = "@python3//:bin/python3",
    requirements_darwin = "//third_party:requirements_darwin.txt",
    requirements_linux = "//third_party:requirements_linux.txt",
)

load("@py_deps//:requirements.bzl", "install_deps")

install_deps()

register_toolchains("//third_party:py_3_toolchain")

#
# rules_sh
#

load("@rules_sh//sh:repositories.bzl", "rules_sh_dependencies")

rules_sh_dependencies()

# Register default toolchains or customize your own.
register_toolchains(
    "@rules_pyo3//pyo3/toolchains:toolchain",
    "@rules_pyo3//pyo3/toolchains:rust_toolchain",
)

nixpkgs_cc_configure(
  name = "gcc",
  repository = "@nixpkgs",
  nix_file_content = "(import <nixpkgs> {}).gcc11",
  cc_std = "gnu++20",
  register = True,
)

register_toolchains("//third_party:nix_cc_python_toolchain")