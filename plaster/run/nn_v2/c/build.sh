#!/usr/bin/env bash

if [[ -z "${DST_FOLDER}" ]]; then
	echo "DST_FOLDER must be defined"
	exit 1;
fi

gcc \
	-c \
	-fpic \
	-O3 \
	./_nn_v2.c \
	-o "${DST_FOLDER}/_nn_v2.o"

gcc \
	-shared \
	-o "${DST_FOLDER}/_nn_v2.so" \
	"${DST_FOLDER}/_nn_v2.o" \
