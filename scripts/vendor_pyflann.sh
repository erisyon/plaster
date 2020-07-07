#!/usr/bin/env bash

##
# This vendors pyflann and applies patches that we've made to it
#
# Git commits are not signed because it's non-trivial to have the
# macOS gpg agent mounted into a linux docker
#
# This script lifts the error helper function from our internal
# repo; at some point we should consolidate these functions

set -eu

_NAME=${0}

trap "exit 1" TERM
export _TOP_PID=$$

error() {
    echo "Error: ${1}"

    # See here https://stackoverflow.com/questions/9893667/is-there-a-way-to-write-a-bash-function-which-aborts-the-whole-execution-no-mat?answertab=active#tab-top
    # to understand this strange kill
    kill -s TERM $_TOP_PID
}

if [[ "${ERISYON_DOCKER_ENV}" != "1" ]]; then
    error "${_NAME} must be run from inside the development container"
fi

if [[ ! -e "plaster_root" ]]; then
    error "${_NAME} must be run from plaster root (file 'plaster_root' not found.)"
fi

# remove existing pyflann prior to revendoring
git rm -rf vendor/pyflann
git commit --no-gpg-sign  -m "remove previously vendored pyflann"

# pip install into vendor/pyflann
pip install -U --target vendor/ pyflann==1.6.14
# pip creates this also, clean it up
rm -rf vendor/pyflann-*-info

# commit pyflann after first stage of vendor, necessary otherwise
# patch will fail due to files not being in the index

git add vendor/pyflann
git commit --no-gpg-sign --amend -m "vendor pyflann" vendor/pyflann

# apply patches
git am --no-gpg-sign vendor/patches/0001-pyflann.patch
