{ callPackage, python }:
rec {
  embree3 = callPackage ./embree3 { };
  open3d = callPackage ./open3d {
    inherit (python.pkgs) buildPythonPackage;
    inherit embree3;
  };
}
