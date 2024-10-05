{
  description = "Mobile Mapping WS2024";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };
  outputs =
    {
      self,
      nixpkgs,
      treefmt-nix,
      ...
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      treefmtEval = treefmt-nix.lib.evalModule pkgs {
        # Used to find the project root
        projectRootFile = "flake.nix";

        programs = {
          black.enable = true;
          isort.enable = true;
          prettier.enable = true;
          nixfmt.enable = true;
        };
      };
      python = pkgs.python3.override {
        packageOverrides = self: super: {
          opencv4 = super.opencv4.override {
            inherit (pkgs) gtk2;
            enableGtk2 = true;
            enableFfmpeg = true;
          };
        };
        self = python;
      };
      pythonEnv = python.withPackages (
        ps: with ps; [
          matplotlib
          numpy
          opencv4
          plotly
          plyfile
        ]
      );
      streamlitRun = pkgs.writeShellApplication {
        name = "streamlitRun";
        runtimeInputs = with pkgs; [
          pythonEnv
          streamlit
        ];
        text = ''${pkgs.streamlit}/bin/streamlit run ./src/MobileMappingWS2024.py'';
      };
    in
    {
      apps.${system} = {
        default = {
          type = "app";
          program = "${streamlitRun}/bin/streamlitRun";
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          treefmtEval.config.build.wrapper
          pythonEnv
          vlc
          streamlit
        ];
      };

      formatter.${system} = treefmtEval.config.build.wrapper;
    };
}
