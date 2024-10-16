{ python, ... }:
python.pkgs.buildPythonPackage {
  pname = "streamlitBackCameraInput";
  version = "0.1.0";
  src = ./.;
  propagatedBuildInputs = [ python.pkgs.streamlit ];
  depenencies = [ ./src/streamlit_back_camera_input/frontend ];
}
