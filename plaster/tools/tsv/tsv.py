"""
These are files that are written by Angela's scope script.
That script is odd in several ways:
    * It is in utf-16
    * It incorrectly encodes tabs (\t) as backslash-t. ie "\\t" as a 2-character string!
    * Same with newlines: "\\n"
"""
from plumbum import local
from plaster.tools.utils import utils
from plaster.tools.log.log import important


def parse_tsv(tsv):
    kvs = {}
    for line in tsv.split("\\n"):
        parts = line.split("\\t")
        if len(parts) == 2:
            # These are in the form:
            # ("part1/part2/part3", "some value")
            # The "channel" is special because it is a LIST so that
            # has to be treated separately.

            k, v = parts
            k = k.replace("/", ".")

            # If v can be converted into a float, great, otherwise leave it alone
            try:
                v = float(v)
            except ValueError:
                pass

            try:
                # Remember if "channel" was present before the exception
                had_channel = "channel" in kvs
                utils.block_update(kvs, k, v)
            except ValueError:
                # This is probably because the "channel" key has
                # not been yet been constructed; add "channel" and try again
                n_channels = int(kvs.get("n_channels", 0))
                if not had_channel and n_channels > 0:
                    kvs["channel"] = [dict() for c in range(n_channels)]

                # RETRY
                utils.block_update(kvs, k, v)
    return kvs


def load_tsv(tsv_path):
    with open(tsv_path, "rb") as f:
        return parse_tsv(f.read().decode("utf-16"))


def load_tsv_for_folder(folder):
    tsv_data = {}
    tsv = sorted(list(local.path(folder) // "*.tsv"))
    if tsv is not None:
        if len(tsv) > 1:
            raise ValueError(f"Too many .tsv files were found in {folder}")
        if len(tsv) == 1:
            try:
                tsv_data = load_tsv(tsv[0])
            except FileNotFoundError:
                pass
            except Exception:
                # Ignore any problems
                important(f"File {tsv[0]} was not readable")
    return tsv_data
