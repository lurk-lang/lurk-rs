{
  description = "Lurk language for zero knowledge proofs";
  inputs = {
    nixpkgs.url = github:nixos/nixpkgs;
    flake-utils = {
      url = github:numtide/flake-utils;
      inputs.nixpkgs.follows = "nixpkgs";
    };
    naersk = {
      url = github:nix-community/naersk;
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self
    , nixpkgs
    , flake-utils
    , naersk
    }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      lib = nixpkgs.lib.${system};
      pkgs = nixpkgs.legacyPackages.${system};
      rustTools = import ./nix/rust.nix {
        nixpkgs = pkgs;
      };
      getRust =
        { channel ? "nightly"
        , date
        , sha256
        , targets ? [ "wasm32-unknown-unknown" "wasm32-wasi" ]
        }: (rustTools.rustChannelOf {
          inherit channel date sha256;
        }).rust.override {
          inherit targets;
          extensions = [ "rust-src" "rust-analysis" ];
        };
      # This is the version used across projects
      rust2022-02-20 = getRust { date = "2022-02-20"; sha256 = "sha256-ZptNrC/0Eyr0c3IiXVWTJbuprFHq6E1KfBgqjGQBIRs="; };
      rust2022-03-15 = getRust { date = "2022-03-15"; sha256 = "sha256-C7X95SGY0D7Z17I8J9hg3z9cRnpXP7FjAOkvEdtB9nE="; };
      rust2022-04-14 = getRust { date = "2022-04-14"; sha256 = "sha256-5sq1QCaKlh84bpGfo040f+zQriJFW7rJO9tZ4rbaQgo="; };
      rust = rust2022-04-14;
      # Get a naersk with the input rust version
      naerskWithRust = rust: naersk.lib."${system}".override {
        rustc = rust;
        cargo = rust;
      };
      # Naersk using the default rust version
      naerskDefault = naerskWithRust rust;
      buildRustProject = pkgs.makeOverridable ({ rust, naersk ? naerskWithRust rust, ... } @ args: naersk.buildPackage ({
        buildInputs = with pkgs; [ ];
        targets = [ ];
        copyLibs = true;
        remapPathPrefix =
          true; # remove nix store references for a smaller output package
      } // args));

      # Convenient for running tests
      testProject = buildRustProject { doCheck = true; inherit rust root; };
      # Load a nightly rust. The hash takes precedence over the date so remember to set it to
      # something like `lib.fakeSha256` when changing the date.
      crateName = "lurk";
      root = ./.;
      buildInputs = with pkgs;
      if !stdenv.isDarwin
      then [ ocl-icd m4 ]
      else [
        darwin.apple_sdk.frameworks.OpenCL
        m4
      ];
      env = {        
        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
        # C_INCLUDE_PATH = "${pkgs.llvmPackages_6.libcxx}/lib";
        # CPP_INCLUDE_PATH = "${pkgs.llvmPackages_6.libcxx}/lib";
        # LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib64:$LD_LIBRARY_PATH";
      };
      # This is a wrapper around naersk build
      # Remember to add Cargo.lock to git for naersk to work
      project = buildRustProject ({
        inherit root rust buildInputs;
      } // env);
      lurk-example = project.override {
        cargoBuildOptions = d: d ++ [ "--example lurk" ];
        copySources = [ "examples" "src" ];
        copyBins = true;
      };
      lurk-wasm = project.override {
        buildInputs = [ pkgs.llvm, pkgs.lld_14 ];
        cargoBuildOptions = d: d ++ [
             "--target"
             "wasm32-unknown-unknown"
             "--features"
             "wasm"
             "--no-default-features"
        ];
        copyTarget = true;
      };
    in
    {
      packages = {
        ${crateName} = project;
        inherit lurk-example lurk-wasm;
        "${crateName}-test" = testProject;
      };

      defaultPackage = self.packages.${system}.${crateName};

      # `nix develop`
      devShell = pkgs.mkShell ({
        inputsFrom = builtins.attrValues self.packages.${system};
        nativeBuildInputs = [ rust ];
        buildInputs = with pkgs; [
          rust-analyzer
          clippy
          rustfmt
          wasm-pack
          nodejs
          yarn
          binaryen
        ];
      } // env);
    });
}
