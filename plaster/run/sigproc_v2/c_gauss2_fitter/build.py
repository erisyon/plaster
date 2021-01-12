#!/usr/bin/env python

from plaster.tools.utils.utils import any_out_of_date
from plumbum import FG, local


def build(dst_folder, c_common_folder):
    with local.cwd("./levmar-2.6"):
        with local.env(ENV_OPTS="-fPIC"):
            local["make"]()

    c_opts = [
        "-c",
        "-fpic",
        "-O3",
        "-I",
        c_common_folder,
        "-I",
        "./levmar-2.6",
        "-DDEBUG",
    ]
    gcc = local["gcc"]

    def build_c(src_name, include_files):
        base_src_name = local.path(src_name).stem
        target_o = f"{dst_folder}/_{base_src_name}.o"
        if any_out_of_date(parents=[src_name, *include_files], children=[target_o],):
            gcc[c_opts, src_name, "-o", target_o] & FG
        return target_o

    common_include_files = [
        f"{c_common_folder}/c_common.h",
    ]
    gauss2_fitter_o = build_c("gauss2_fitter.c", common_include_files)
    c_common_o = build_c(f"{c_common_folder}/c_common.c", common_include_files)

    gauss2_fitter_so = f"{dst_folder}/_gauss2_fitter.so"
    if any_out_of_date(
        parents=[gauss2_fitter_o, c_common_o], children=[gauss2_fitter_so]
    ):
        gcc[
            "-shared",
            "-o",
            gauss2_fitter_so,
            gauss2_fitter_o,
            c_common_o,
            "-L",
            "./levmar-2.6",
            "-llevmar",
            "-lm",
            "-llapack",
            "-lblas",
        ] & FG


if __name__ == "__main__":
    build(
        dst_folder="/erisyon/plaster/plaster/run/sigproc_v2/c_gauss2_fitter",
        c_common_folder="/erisyon/plaster/plaster/tools/c_common",
    )
