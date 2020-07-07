#!/usr/bin/env bash

# This is called by the git filter that is installed with:
# git config --local filter.strip_notebook.smudge cat
# git config --local filter.strip_notebook.clean ./plaster/scripts/strip_notebook.sh
# git config --local filter.strip_notebook.required true

if [[ ! -x $(command -v "jq") ]]; then
	cat <&0
else
	# Use jq to strip output
	# echo "Stripping notebooks outputs" >&2
	jq --indent 1 '(.cells[] | select(has("outputs")) | .outputs) = []'
fi
