{ callPackage, python }:
rec {
  streamlitBackCameraInput = callPackage ./streamlit_back_camera_input { inherit python; };
  embree3 = callPackage ./embree3 { };
  open3d = callPackage ./open3d {
    inherit (python.pkgs) buildPythonPackage;
    inherit embree3;
  };
}
