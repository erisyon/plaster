#!/usr/bin/env bash

if [[ -z "${DST_FOLDER}" ]]; then
	echo "DST_FOLDER must be defined"
	exit 1;
fi

if [[ -z "${C_COMMON_FOLDER}" ]]; then
	echo "C_COMMON_FOLDER must be defined"
	exit 1;
fi

C_OPTS="-c -fpic -O3 -I ${C_COMMON_FOLDER}"

gcc \
	$C_OPTS \
	./_nn_v2.c \
	-o "${DST_FOLDER}/_nn_v2.o" \

gcc \
	$C_OPTS \
	"${C_COMMON_FOLDER}/c_common.c" \
	-o "${DST_FOLDER}/c_common.o" \

gcc \
	-shared \
	-o "${DST_FOLDER}/_nn_v2.so" \
	"${DST_FOLDER}/_nn_v2.o" \
	"${DST_FOLDER}/c_common.o" \
