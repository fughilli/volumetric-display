"""Provides a function to register packages from Nix for use in Bazel.

"""

load("@io_tweag_rules_nixpkgs//nixpkgs:nixpkgs.bzl", "nixpkgs_package")

standard_packages = [
    "glm",
    "boost",
    "glew",
    "glfw",
    "chuck",
]

packages_build_files = {
    "glm": "//third_party:glm.BUILD",
    "glew": "//third_party:glew.BUILD",
    "glfw": "//third_party:glfw.BUILD",
    "boost": "//third_party:boost.BUILD",
}

packages_nix_files = {
    "glew": "//third_party:glew.nix",
}

def register_packages():
    """Registers the standard packages with the nixpkgs repository."""
    for package in standard_packages:
        kwargs = {}
        if package in packages_build_files:
            kwargs["build_file"] = packages_build_files[package]
        if package in packages_nix_files:
            kwargs["nix_file"] = packages_nix_files.get(package)
        else:
            kwargs["nix_file_content"] = """
        with import <nixpkgs> {{}};
        let pkg = {package};
        in symlinkJoin {{
          name = "{package}-joined";
          paths = [
            pkg
          ] ++ lib.optionals (builtins.hasAttr "dev" pkg) [
            pkg.dev
          ] ++ lib.optionals (builtins.hasAttr "bin" pkg) [
            pkg.bin
          ];
        }}
        """.format(package = package)
        nixpkgs_package(
            name = package,
            repositories = {"nixpkgs": "@nixpkgs"},
            **kwargs
        )

    nixpkgs_package(
        name = "python3",
        repositories = {"nixpkgs": "@nixpkgs"},
        # netifaces fails to build in the nixpkgs environment on macOS, so we
        # bring it in via nixpkgs.withPackages.
        nix_file_content = """
          with import <nixpkgs> {};
          let pkg = python3.withPackages (ps: with ps; [
            netifaces
          ]);
          in pkg
          #symlinkJoin {
          #  name = "python3-with-dev";
          #  paths = [
          #    pkg pkg.dev
          #  ];
          #}
        """,
    )
