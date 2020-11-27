#!/usr/bin/env python

from plumbum import local, FG
from plaster.tools.utils.utils import any_out_of_date


def build(dst_folder, c_common_folder):
    c_opts = [
        "-c",
        "-fpic",
        "-O3",
        "-I",
        c_common_folder,
        "-DDEBUG",
    ]
    gcc = local["gcc"]

    def build_c(src_name, include_files):
        base_src_name = local.path(src_name).stem
        target_o = f"{dst_folder}/_{base_src_name}.o"
        if any_out_of_date(parents=[src_name, *include_files], children=[target_o],):
            gcc[c_opts, src_name, "-o", target_o] & FG
        return target_o

    common_include_files = [f"{c_common_folder}/c_common.h"]
    radiometry_o = build_c("radiometry.c", common_include_files)
    c_common_o = build_c(f"{c_common_folder}/c_common.c", common_include_files)

    radiometry_so = f"{dst_folder}/_radiometry.so"
    if any_out_of_date(parents=[radiometry_o, c_common_o], children=[radiometry_so]):
        gcc["-shared", "-o", radiometry_so, radiometry_o, c_common_o, "-lm",] & FG


if __name__ == "__main__":
    build(
        dst_folder="/erisyon/plaster/plaster/run/sigproc_v2/c_radiometry",
        c_common_folder="/erisyon/plaster/plaster/tools/c_common",
    )
