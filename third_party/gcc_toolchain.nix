# cross compiler toolchain for cc rules in Bazel
# execution platform: aarch64-darwin
# target platform: x86_64-linux
# inspired by https://github.com/tweag/rules_nixpkgs/blob/21c4ea481021cb51a6e5d0969b2cee03dba5a637/examples/toolchains/cc_cross_osx_to_linux_amd64/toolchains/osxcross_cc.nix
let
  og = import <nixpkgs> { };
  nixpkgs = import <nixpkgs> {
    buildSystem = builtins.currentSystem;
    hostSystem = builtins.currentSystem;
  };
in
let
  pkgs = nixpkgs.buildPackages;
in
pkgs.buildEnv (
  let
    cc = pkgs.llvmPackages_11.clang;
  in
  {
    name = "bazel-${cc.name}-cc";
    paths = [ cc ];
    pathsToLink = [ "/bin" ];
    passthru = {
      inherit (cc) isClang targetPrefix;
      orignalName = cc.name;
    };
  }
)
