#!/usr/bin/env bash

# This is the entrypoint for e2e tests which test environmental setup
# and are therefore run _outside_ of the docker container.

if [[ "${ERISYON_DOCKER_ENV}" == "1" ]]; then
	echo "Error: E2E test should be run outside of the docker environment"
	exit 1
fi

USE_DOCKER_CACHE=${USE_DOCKER_CACHE:-"1"}
_DOCKER_CACHE=""
if [[ "${USE_DOCKER_CACHE}" != "1" ]]; then
	_DOCKER_CACHE="--no-cache"
fi

NoColor='\033[0m'
Black='\033[0;30m'
Red='\033[0;31m'
Green='\033[0;32m'
Yellow='\033[0;33m'
Blue='\033[0;34m'
Purple='\033[0;35m'
Cyan='\033[0;36m'
White='\033[0;37m'
BBlack='\033[1;30m'
BRed='\033[1;31m'
BGreen='\033[1;32m'
BYellow='\033[1;33m'
BBlue='\033[1;34m'
BPurple='\033[1;35m'
BCyan='\033[1;36m'
BWhite='\033[1;37m'

set -euo pipefail
trap "exit 1" TERM
export _TOP_PID=$$
_error() {
    printf "\n!!!! ${BRed}ERROR in e2e.sh${NoColor} !!!!\n"
    printf "${Red}${@}${NoColor}\n\n"

    # See here https://stackoverflow.com/questions/9893667/is-there-a-way-to-write-a-bash-function-which-aborts-the-whole-execution-no-mat?answertab=active#tab-top
    # to understand this strange kill
    kill -s TERM $_TOP_PID
}

_major_separator() {
	printf "${BYellow}----------------------------------------------------------------------------------------------------------${NoColor}\n"
}

_run() {
	type -t "$1" | grep "function" > /dev/null
	if [[ "$?" == "0" ]]; then
		printf "Running : ${BYellow}${1}${NoColor}\n"
		$1
		return $?
	else
		printf "Skipping: ${Yellow}${1}${NoColor}\n"
		return 0
	fi
}

_check_plaster_root() {
	if [[ ! -e "plaster_root" ]]; then
		_error "Must be run from plaster_root"
	fi
}

_fail_test() {
	_error "TEST FAILED LINE ${1}"
}

_check_fail_test() {
	[[ "${1}" == "0" ]] || _fail_test "${2}"
}

_sleep() {
	echo "Sleeping: $1 seconds..."
	sleep $1
}

_check_plaster_root


# plaster container tests
# ------------------------------------------------------------------------------

test_plaster_docker_container() {
	_check_plaster_root

	# Note: This is testing plaster build directly without docker_build.sh
	docker build ${_DOCKER_CACHE} -t "plaster:e2e" .

	DOCKER_RUN="docker run -it --volume ${HOME}/jobs_folder:/jobs_folder:rw plaster:e2e"

	it_passes_tests() {
		LIMIT_TESTS=""
		$DOCKER_RUN plas test "${LIMIT_TESTS}" --no_clear
		_check_fail_test $? $LINENO
	}

	it_gens_and_runs() {
		rm -rf ~/jobs_folder/__e2e_test
		$DOCKER_RUN \
			plas gen classify \
			--job=/jobs_folder/__e2e_test \
			--sample=e2e_test \
		    --decoys=none \
    		--protein_seq="pep25:GCAGCAGAG" \
    		--n_pres=0 \
		    --n_edmans=8 \
		    --label_set='C'
		_check_fail_test $? $LINENO

		$DOCKER_RUN plas run /jobs_folder/__e2e_test
		_check_fail_test $? $LINENO
	}

	it_starts_plaster_container_as_non_dev() {
		docker run -it plaster:e2e plas gen --readme | grep -i "GEN -- The plaster run generator" > /dev/null
		_check_fail_test $? $LINENO

		echo 1 > __e2e_test_file
		$DOCKER_RUN plas bash ls -l | grep -i "__e2e_test_file" > /dev/null
		[[ "$?" == "1" ]] || _check_fail_test $? $LINENO
		rm __e2e_test_file
	}

	it_runs_jupyter() {
		rm -f ./scripts/e2e_test_notebook.html
		cp ./scripts/e2e_test_notebook.ipynb ~/jobs_folder/__e2e_test
		$DOCKER_RUN jupyter nbconvert --to html --execute /jobs_folder/__e2e_test/e2e_test_notebook.ipynb > /dev/null
		grep -q -i "successfulrun" ~/jobs_folder/__e2e_test/e2e_test_notebook.html
		_check_fail_test $? $LINENO
		rm -f ./scripts/e2e_test_notebook.html
	}

	_run "it_passes_tests" || _fail_test $LINENO
	_run "it_gens_and_runs" || _fail_test $LINENO
	_run "it_starts_plaster_container_as_non_dev" || _fail_test $LINENO
	_run "it_runs_jupyter" || _fail_test $LINENO
}

_major_separator
_run "test_plaster_docker_container" || _fail_test $LINENO
