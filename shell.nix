{ pkgs ? import <nixpkgs> {}}:

pkgs.poetry2nix.mkPoetryEnv {
    projectDir = ./.;
}

# pkgs.mkShell {
#   nativeBuildInputs = [
#     pkgs.python38
#     pkgs.poetry
#     pkgs.python
#   ];
#   shellHook = ''
#     # create venv if it doesn't exist
#     poetry run true
#
#     export VIRTUAL_ENV=$(poetry env info --path)
#     export POETRY_ACTIVE=1
#     source "$VIRTUAL_ENV/bin/activate"
#   '';
# }
