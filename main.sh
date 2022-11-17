#!/bin/bash

nix-shell --run "python -m typer illixr.analysis.main main ${@}"
