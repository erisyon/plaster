from plaster.tools.utils.utils import any_out_of_date
from plumbum import FG, local


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

    def build_c(src_name, include_files):
        base_src_name = local.path(src_name).stem
        target_o = f"{dst_folder}/_{base_src_name}.o"
        if any_out_of_date(parents=[src_name, *include_files], children=[target_o],):
            gcc[c_opts, src_name, "-o", target_o] & FG
        return target_o

    common_include_files = [
        f"{c_common_folder}/c_common.h",
    ]
    survey_v2_o = build_c("survey_v2.c", common_include_files)
    c_common_o = build_c(f"{c_common_folder}/c_common.c", common_include_files)

    survey_v2_so = f"{dst_folder}/_survey_v2.so"
    if any_out_of_date(parents=[survey_v2_o, c_common_o], children=[survey_v2_so]):
        gcc[
            "-shared",
            survey_v2_o,
            c_common_o,
            "-L",
            flann_lib_folder,
            "-lflann",
            "-o",
            survey_v2_so,
        ] & FG


if __name__ == "__main__":
    build(
        dst_folder="/erisyon/plaster/plaster/run/survey_v2/c",
        c_common_folder="/erisyon/plaster/plaster/tools/c_common",
        flann_include_folder="/flann/src/cpp/flann",
        flann_lib_folder="/flann/lib",
    )
