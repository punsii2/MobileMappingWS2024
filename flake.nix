{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      python = pkgs.python3.override {
        packageOverrides = self:
          super: {
            opencv4 = super.opencv4.override {
              inherit (pkgs) gtk2;
              enableGtk2 = true;
              enableFfmpeg = true; #here is how to add ffmpeg and other compilation flags
            };
          };
        self = python;
      };
      pythonEnv = python.withPackages (ps: with ps; [
        black
        matplotlib
        numpy
        opencv4
        pip
        virtualenv
      ]);
      pythonApp = pkgs.python3Packages.buildPythonPackage {
        name = "readstray";
        src = ./src;
        propagatedBuildInputs = [ pythonEnv ];
        doCheck = false;
      };
    in
    {
      packages.${system} = {
        inherit pythonApp;
        default = pythonApp;
      };

      devShells.${system}.default =
        pkgs.mkShell {
          packages = with pkgs;
            [
              pythonEnv
              treefmt
              vlc
              nixpkgs-fmt
              streamlit
            ];
        };

      formatter.${system} = pkgs.nixpkgs-fmt;
    };
}
