"""
Helpers for reading proteins in various forms.

These are typically helpers that are needed by both GENERATORS and PGEN.

If the method is only needed by GENERATORS then it should be added to
the BaseGenerator class.
"""

import time
import pandas as pd
import numpy as np
from plaster.tools.utils.simple_http import http_method
import hashlib
from io import StringIO
from munch import Munch
from plumbum import local, FG
from plaster.tools.schema import check
from plaster.tools.utils import utils
from plaster.tools.uniprot import uniprot
from plaster.tools.log import log
from plaster.tools.log.log import debug, info, important
from plaster.gen.errors import ValidationError


def _url_get(url):
    return http_method(url)


def cache_source(cache_folder, source, copy_to):
    """
    If this is a URL or S3 fetch and cache at cache_folder.  Local files
    can be loaded from //jobs_folder/... only.
    In all cases, the cached file is optionally copied to copy_to so that
    that job folders contain a copy of all gen source data.

    Returns:
        the contents of the file
    """

    def make_cache_path():
        local.path(cache_folder).mkdir()
        return (
            local.path(cache_folder) / hashlib.md5(source.encode("utf-8")).hexdigest()
        )

    file_contents = None
    if source.startswith("http://") or source.startswith("https://"):
        cache_path = make_cache_path()
        if not cache_path.exists():
            log.info(f"Fetching {source}... (TO {cache_path})")
            file_contents = _url_get(source)
            utils.save(cache_path, file_contents)
        else:
            file_contents = utils.load(cache_path)
    elif source.startswith("s3://"):
        cache_path = make_cache_path()
        if not cache_path.exists():
            important(f"Syncing from {source} to {cache_path}")
            local["aws"]["s3", "cp", source, cache_path] & FG
            # s3 cp already saves it to disk, fall thru & load
        file_contents = utils.load(cache_path)
    else:
        file_contents = utils.load(source)

    assert file_contents is not None

    if copy_to:
        assert local.path(copy_to).exists()
        filename = local.path(source).basename
        utils.save(copy_to / filename, file_contents)

    return file_contents


def protein_fasta(fasta_str, override_name=None):
    """
    Parse fasta format as defined by Uniprot.
    https://www.uniprot.org/help/fasta-headers

    Arguments:
        fasta_str: The string (not the file path!) of the fasta
        override_name: If non-None will replace the found protein name

    Returns:
        List(Dict(id, seqstr))
    """

    seqs = uniprot.fasta_split(fasta_str)
    ret = []

    for header, seqstr in seqs:
        header = header.split(" ")[0]
        parts = header.split("|")
        if len(parts) >= 2:
            id = parts[1]
        else:
            id = parts[0]

        if override_name is not None:
            id = override_name
            assert len(seqs) == 1

        ret += [dict(id=id, seqstr=seqstr)]

    return ret


def split_protein_name(identifier):
    # Example: Name:p123 vs
    parts = identifier.split(":")
    if len(parts) == 1:
        return dict(seqstr=parts[0], id=None)
    elif len(parts) == 2:
        return dict(id=parts[0], seqstr=parts[1])
    raise ValidationError("names must follow Name=Seq convention.")


def _uniprot_lookup(ac):
    """Mock-point"""
    fasta = uniprot.get_ac_fasta(ac)
    return protein_fasta(fasta)


def _protein_csv_warning(warning_string):
    """Mock-point"""
    important(warning_string)


def _protein_csv_info(info_string):
    """Mock-point"""
    info(info_string)


def protein_csv_df(csv_string):
    """
    Parse protein(s) in csv format.

    Must have a header row with some of these column names:
        'Name', 'Seq', 'UniprotAC', 'Abundance', 'PTM', 'POI', columns.

    If Name is present that will become the name
    else and UniprotAC is present then that will become the name

    If Seq is absent and UniprotAC is present then it
    will pull the seqs from Uniprot

    If Abundance is present it will be returned as a parallel array

    If PTM is present it will be added into the protein df

    If POI is present it will be added into the protein df (pro_in_report)

    Returns:
        DataFrame: (id, seqstr, abundance, ptm_locs)
    """
    src_df = pd.read_csv(StringIO(csv_string))
    for col in src_df.columns:
        src_df = src_df.rename(columns={col: col.strip()})

    if "Seq" not in src_df and "UniprotAC" not in src_df:
        raise ValueError("protein csv missing either a Seq or a UniprotAC column")

    if "Seq" in src_df and "UniprotAC" in src_df:
        raise ValueError("protein csv has both a Seq and a UniprotAC column")
    if "UniprotAC" in src_df.columns:
        # Using the UniprotAC as the Seq column
        _protein_csv_info(f"Requesting ACs from uniprot, this may take a while")
        dst_df = pd.DataFrame(dict(id=src_df.UniprotAC, seqstr=""))
        start_time = time.time()
        for i, ac in enumerate(dst_df.id):
            if time.time() - start_time > 10.0:
                info(
                    f"{i} of {len(dst_df.id)}. If it blocks, consider a ^C and restart."
                )
            seqs = _uniprot_lookup(ac)
            n_seqs = len(seqs)
            seqstr = None
            if n_seqs == 0:
                _protein_csv_warning(f"Uniprot ac {ac} returned no sequences. Ignoring")
            elif n_seqs > 1:
                _protein_csv_warning(
                    f"Uniprot ac {ac} returned > 1 sequence. Using the longest"
                )
                longest = 0
                for seq in seqs:
                    if len(seq["seqstr"]) > longest:
                        seqstr = seq["seqstr"]
                        longest = len(seqstr)
            else:
                seqstr = seqs[0]["seqstr"]

            if seqstr is not None:
                dst_df.loc[i, "seqstr"] = seqstr

        if "Name" in src_df:
            # Overload the UniprotAC with the Name
            dst_df["id"] = src_df.Name

    else:
        # Using the Seq column
        if "Name" not in src_df:
            raise ValueError(
                "protein csv missing a Name column without a UniprotAC column"
            )

        dst_df = pd.DataFrame(dict(id=src_df.Name, seqstr=src_df.Seq))

    # ADD the PTM column if present
    if "PTM" in src_df:
        dst_df["ptm_locs"] = src_df.fillna("").PTM.astype(str)
    else:
        dst_df["ptm_locs"] = ""

    # ADD the abundance column if present
    if "Abundance" in src_df:
        dst_df["abundance"] = src_df.Abundance.astype(float)
    else:
        dst_df["abundance"] = np.nan

    # ADD the POI column if present; note that the gen --protein_of_interest flag may
    # cause this to be overridden.
    if "POI" in src_df:
        dst_df["in_report"] = src_df.fillna(0).POI.astype(int)

    # STRIP whitespace
    dst_df["id"] = dst_df.id.astype(str).apply(lambda x: x.strip())
    dst_df["seqstr"] = dst_df.seqstr.astype(str).apply(lambda x: x.strip())
    dst_df["ptm_locs"] = dst_df.ptm_locs.astype(str).apply(lambda x: x.strip())

    if "Abundance" in src_df:
        dst_df = dst_df.sort_values(["abundance", "id"], ascending=False)
    else:
        dst_df = dst_df.sort_values(["id"])

    dupes = dst_df.id.duplicated(keep=False)
    if dupes.sum() > 0:
        raise ValueError(f"duplicate names in protein_csv:\n{dst_df[dupes]}")

    dupes = dst_df.seqstr.duplicated(keep=False)
    if dupes.sum() > 0:
        raise ValueError("duplicate seqs in protein_csv")

    return dst_df.reset_index()


def task_rename(task_block, new_name):
    """Expects a task dictionary that has only one root key, then renames that key"""
    check.t(task_block, Munch)
    old_name = utils.get_root_key(task_block)
    task_block[old_name].task = old_name
    utils.ren_key(task_block, old_name, new_name)
