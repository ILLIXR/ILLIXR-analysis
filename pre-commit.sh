#!/bin/sh

poetry run typer illixr.analysis.main run check --no-modify
ret="${?}"
if [ "${ret}" -ne 0 ]; then
	echo "Did you forget to run 'typer illixr.analysis.main run check'?"
	echo "Run 'git commit --no-verify' to skip this check."
	exit "${ret}"
fi
