with import <nixpkgs> { };
let
  pkg = glfw;
in
symlinkJoin {
  name = "glfw-joined";
  paths = [
    pkg
  ] ++ lib.optionals (builtins.hasAttr "dev" pkg) [
    pkg.dev
  ] ++ lib.optionals (builtins.hasAttr "bin" pkg) [
    pkg.bin
  ];
  postBuild = ''
    # Create symlink libglfw3.dylib -> libglfw.3.3.dylib for glfw-sys compatibility
    # glfw-sys build script expects libglfw3.dylib but nixpkgs provides libglfw.3.3.dylib
    if [ -f "$out/lib/libglfw.3.3.dylib" ] && [ ! -f "$out/lib/libglfw3.dylib" ]; then
      ln -s libglfw.3.3.dylib "$out/lib/libglfw3.dylib"
    fi
    # Also create symlink for Linux (.so files)
    if [ -f "$out/lib/libglfw.so.3.3" ] && [ ! -f "$out/lib/libglfw3.so" ]; then
      ln -s libglfw.so.3.3 "$out/lib/libglfw3.so"
    fi
  '';
}
