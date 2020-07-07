#!/usr/bin/env bash

# This is called by the git filter that is installed with:
# git config --local filter.black.smudge cat
# git config --local filter.black.clean ./plaster/scripts/black.sh
# git config --local filter.black.required true

if [[ ! -x $(command -v "black") ]]; then
	# If black isn't avail, no problem
	cat <&0
else
	set +e  # Allow black to fail which will cause cat to print the unchanged file back to stdout
	SRC=$(mktemp)
	cp /dev/stdin $SRC
	black $SRC 2> /dev/null
	cat $SRC
fi
