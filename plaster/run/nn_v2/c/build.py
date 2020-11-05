from plumbum import local, FG
from plaster.tools.utils.utils import any_out_of_date


def build(dst_folder, c_common_folder, flann_include_folder, flann_lib_folder):
    c_opts = [
        "-c",
        "-fpic",
        "-O3",
        "-I",
        c_common_folder,
        "-I",
        flann_include_folder,
        "-DDEBUG",
    ]
    gcc = local["gcc"]

    nn_v2_o = f"{dst_folder}/_nn_v2.o"
    if any_out_of_date(
        parents=["./_nn_v2.c", "./_nn_v2.h", f"{c_common_folder}/c_common.h"],
        children=[nn_v2_o],
    ):
        gcc[c_opts, "./_nn_v2.c", "-o", nn_v2_o] & FG

    c_common_o = f"{dst_folder}/c_common.o"
    if any_out_of_date(
        parents=[f"{c_common_folder}/c_common.h", f"{c_common_folder}/c_common.c"],
        children=[c_common_o],
    ):
        gcc[c_opts, f"{c_common_folder}/c_common.c", "-o", c_common_o] & FG

    nn_v2_so = f"{dst_folder}/_nn_v2.so"
    if any_out_of_date(parents=[nn_v2_o, c_common_o], children=[nn_v2_so]):
        gcc[
            "-shared",
            nn_v2_o,
            c_common_o,
            "-L",
            flann_lib_folder,
            "-lflann",
            "-o",
            nn_v2_so,
        ] & FG


if __name__ == "__main__":
    build(
        dst_folder="/erisyon/plaster/plaster/run/nn_v2/c",
        c_common_folder="/erisyon/plaster/plaster/tools/c_common",
        flann_include_folder="/flann/src/cpp/flann",
        flann_lib_folder="/flann/lib",
    )
