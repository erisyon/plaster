from plumbum import local, FG


def build(dst_folder, c_common_folder, flann_include_folder, flann_lib_folder):
    c_opts = ["-c", "-fpic", "-O3", "-I", c_common_folder, "-I", flann_include_folder]
    gcc = local["gcc"]

    gcc[c_opts, "./_nn_v2.c", "-o", f"{dst_folder}/_nn_v2.o"] & FG
    gcc[c_opts, f"{c_common_folder}/c_common.c", "-o", f"{dst_folder}/c_common.o"] & FG
    gcc[
        "-shared",
        "-o",
        f"{dst_folder}/_nn_v2.so",
        f"{dst_folder}/_nn_v2.o",
        f"{dst_folder}/c_common.o",
        "-L",
        flann_lib_folder,
        "-lflann",
    ] & FG
