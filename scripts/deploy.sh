#!/usr/bin/env bash


if [[ ! -e "plaster_root" ]]; then
	echo "Are you sure you are in the plaster folder? (plaster_root not found.)"
	exit 1
fi

# BUMP version
VERSION=$(cat ./plaster/version.py | sed 's/__version__[ ]*=[ ]*\"//')
VERSION=$(echo $VERSION | sed 's/\"//g')
NEW_VERSION=$(echo $VERSION | awk -F. '{$NF = $NF + 1;} 1' | sed 's/ /./g')

echo "Bumping from version $VERSION to $NEW_VERSION.  ENTER to continue or ^c."
read -p "$*"

echo "__version__ = \"${NEW_VERSION}\"" > ./plaster/version.py

# Clean previous runs
rm -rf ./erisyon.plaster.egg-info

pipenv run python setup.py sdist \
	&& pipenv run twine check dist/* \
	&& pipenv run twine upload dist/* \
	&& rm -rf ./erisyon.plaster.egg-info
