#!/usr/bin/env python

from plaster.tools.utils.utils import any_out_of_date
from plumbum import FG, local


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
        if (
            any_out_of_date(parents=[src_name, *include_files], children=[target_o],)
            or True
        ):
            gcc[c_opts, src_name, "-o", target_o] & FG
        return target_o

    common_include_files = [
        f"{c_common_folder}/c_common.h",
    ]
    o_filename = build_c(f"{dst_folder}/sim_v2.c", common_include_files)
    c_common_o = build_c(f"{c_common_folder}/c_common.c", common_include_files)

    so_filename = f"{dst_folder}/_sim_v2.so"
    if (
        any_out_of_date(parents=[o_filename, c_common_o], children=[so_filename])
        or True
    ):
        gcc["-shared", "-o", so_filename, o_filename, c_common_o, "-lm",] & FG


if __name__ == "__main__":
    build(
        dst_folder="/erisyon/plaster/plaster/run/sim_v2/c",
        c_common_folder="/erisyon/plaster/plaster/tools/c_common",
    )
