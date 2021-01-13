#!/usr/bin/env python

from plaster.tools.utils.utils import any_out_of_date
from plumbum import FG, local


def build(dst_folder, c_common_folder):

    # The CSA spline code is not working well yet and is not being used.
    # It's role is currently filled by sampling the RegPSF at high
    # resolution. We can either fix it or deprecate and remove it
    # and rely on the RegPSF to fill the role in the long term.

    c_opts = [
        "-c",
        "-fpic",
        "-O3",
        "-I",
        c_common_folder,
        "-I",
        "./csa_spline",
        "-DDEBUG",
    ]
    gcc = local["gcc"]

    def build_c(src_name, include_files=tuple()):
        base_src_name = local.path(src_name).stem
        target_o = f"{dst_folder}/_{base_src_name}.o"
        if any_out_of_date(parents=[src_name, *include_files], children=[target_o],):
            gcc[c_opts, src_name, "-o", target_o] & FG
        return target_o

    include_files = (
        f"{c_common_folder}/c_common.h",
        f"./csa_spline/csa.h",
    )
    radiometry_o = build_c("radiometry.c", include_files)
    c_common_o = build_c(f"{c_common_folder}/c_common.c", include_files)
    csa_o = build_c("./csa_spline/csa.c")
    svd_o = build_c("./csa_spline/svd.c")

    radiometry_so = f"{dst_folder}/_radiometry.so"
    if any_out_of_date(parents=[radiometry_o, c_common_o], children=[radiometry_so]):
        gcc[
            "-shared",
            "-o",
            radiometry_so,
            csa_o,
            svd_o,
            radiometry_o,
            c_common_o,
            "-lm",
        ] & FG


if __name__ == "__main__":
    build(
        dst_folder="/erisyon/plaster/plaster/run/sigproc_v2/c_radiometry",
        c_common_folder="/erisyon/plaster/plaster/tools/c_common",
    )
