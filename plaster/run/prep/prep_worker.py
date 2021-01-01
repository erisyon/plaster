from plumbum import local
import numpy as np
from numpy import random
import pandas as pd
import itertools
from plaster.tools.aaseq.aaseq import aa_str_to_list
from plaster.tools.aaseq import proteolyze
from plaster.run.prep.prep_result import PrepResult
from plaster.tools.log.log import debug, info, prof, important, error

# from plaster.tools.parallel_map.parallel_map import (
#     parallel_groupby_apply,
#     parallel_array_split_map,
# )
from plaster.tools.zap import zap

from plaster.tools.schema import check


def _error(msg):
    """Mock-point"""
    error(msg)


def _proteolyze(pro_seq_df, proteases, include_misses=0, limit=None):
    """
    Break a single protein (one aa per row of pro_seq_df) into peptide fragments
    using the proteases, which may be None, a single protease name, or a list of
    protease names.

    If include_misses > 0 then it will include 1 or more missed cleavages.

    This is called by an groupby("pro_i").apply(...)

    Returns:
        A DF where each row is:
            (aa, pro_pep_i, pep_offset_in_pro)
    """

    if include_misses > 0:
        # TASK: Implement
        # Note: take care when multiple proteases
        raise NotImplementedError

    if limit is not None:
        # TASK: Implement
        raise NotImplementedError

    n_aas = len(pro_seq_df)

    if proteases is not None:
        if type(proteases) is not list:
            proteases = [proteases]
        cleave_before_iz = []
        for protease in proteases:
            assert protease is not None
            rules = proteolyze.compile_protease_rules(protease)
            cleave_before_iz += proteolyze.cleavage_indices_from_rules(
                pro_seq_df, rules
            )
        cleave_before_iz = sorted(list(set(cleave_before_iz)))
        start_iz = np.append([0], cleave_before_iz).astype(int)
        stop_iz = np.append(cleave_before_iz, [n_aas]).astype(int)
    else:
        start_iz = np.array([0]).astype(int)
        stop_iz = np.array([n_aas]).astype(int)

    # pro_i = pro_seq_df.pro_i.iloc[0]
    rows = [
        (pro_seq_df.aa.values[offset], pep_i, offset)
        for pep_i, (start_i, stop_i) in enumerate(zip(start_iz, stop_iz))
        for offset in range(start_i, stop_i)
    ]
    return pd.DataFrame(rows, columns=["aa", "pro_pep_i", "pep_offset_in_pro"])


def _step_1_check_for_uniqueness(pro_spec_df):
    # STEP 1: Check for uniqueness
    dupe_seqs = pro_spec_df.sequence.duplicated(keep="first")
    if dupe_seqs.any():
        _error("The following sequences are duplicated")
        for d in pro_spec_df[dupe_seqs].itertuples():
            _error(f"{d.name}={d.sequence}")
        raise ValueError("Duplicate protein seq(s)")

    dupe_names = pro_spec_df.name.duplicated(keep="first")
    if dupe_names.any():
        # TASK: Make a better error enumerating the duplicates
        raise ValueError("Duplicate protein name(s)")


def _step_2_create_pros_and_pro_seqs_dfs(pro_spec_df):
    """
    Create pros_df and pro_seqs_df.
    Converts the sequence as a string into normalzied DataFrames
    """

    # Sort proteins such that the protein(s) being 'reported' are at the top, which means
    # the most interesting peptides start at pep_i==1.
    _pro_spec_df = pro_spec_df.sort_values(by=["report", "name"], ascending=False)

    # pro_lists = parallel_array_split_map(
    #     aa_str_to_list, dict(seqstr=_pro_spec_df.sequence.values)
    # )
    pro_lists = zap.arrays(aa_str_to_list, dict(seqstr=_pro_spec_df.sequence.values))

    # Make a full-df with columns "aa", "pro_i", "pro_name", and "ptm_locs", "pro_report"
    # Then split this into the two fully normalized dfs
    df = pd.DataFrame(
        [
            (i, pro_i + 1, pro_name, pro_ptm_locs, pro_report)
            for pro_i, (pro, pro_name, pro_ptm_locs, pro_report) in enumerate(
                zip(
                    pro_lists,
                    _pro_spec_df.name,
                    _pro_spec_df.ptm_locs,
                    _pro_spec_df.report,
                )
            )
            for i in pro
        ],
        columns=["aa", "pro_i", "pro_name", "pro_ptm_locs", "pro_report"],
    )

    # ADD reserved nul row
    nul = pd.DataFrame(
        [dict(aa=".", pro_i=0, pro_name="nul", pro_ptm_locs="", pro_report=0)]
    )
    df = pd.concat((nul, df))

    pros_df = (
        df[["pro_i", "pro_name", "pro_ptm_locs", "pro_report"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .rename(columns=dict(pro_name="pro_id"))
    )
    pros_df["pro_is_decoy"] = False

    pro_seqs_df = df[["pro_i", "aa"]].reset_index(drop=True)

    return pros_df, pro_seqs_df


def _step_3_generate_decoys(pros_df, pro_seqs_df, decoy_mode):

    decoy_modes = [None, "none", "reverse", "shuffle"]
    if decoy_mode not in decoy_modes:
        debug(decoy_mode)
        raise NotImplementedError

    if decoy_mode is None or decoy_mode == "none":
        return (
            pd.DataFrame([], columns=PrepResult.pros_columns),
            pd.DataFrame([], columns=PrepResult.pro_seqs_columns),
        )

    # decoy_mode is now shuffle or reverse
    decoy_modes = ["reverse", "shuffle"]
    assert decoy_mode in decoy_modes

    smallest_pro_i = pros_df.pro_i.min()
    assert smallest_pro_i == 0

    n_pros = pros_df.pro_i.max() + 1

    # Skip the nul
    pros_df = pros_df.set_index("pro_i").loc[1:]
    pro_seqs_df = pro_seqs_df.set_index("pro_i").loc[1:]

    # -------------------------------------------------------------------------
    # local fns to transform each sequence into a decoy
    def reverse_seq(x):
        x["aa"] = x["aa"].iloc[::-1].values
        return x

    def shuffle_seq(x):
        seq = x["aa"].values.copy()
        np.random.shuffle(seq)
        x["aa"] = seq
        return x

    decoy_transform = reverse_seq if decoy_mode == "reverse" else shuffle_seq
    # -------------------------------------------------------------------------

    decoy_seqs_df = (
        pro_seqs_df.groupby("pro_i", sort=False).apply(decoy_transform).reset_index()
    )
    decoy_seqs_df.pro_i += n_pros - 1  # -1 because we are not counting the nul

    pro_seqs_df = pro_seqs_df.reset_index()
    pros_df = pros_df.reset_index()

    # -------------------------------------------------------------------------
    # local fns to similarly transform PTM locations for decoys
    def reverse_pro_ptm_locs():
        reversed_locs = []
        for pro in pros_df.itertuples():
            seq_length = pro_seqs_df[pro_seqs_df.pro_i == pro.pro_i].counts()[0]
            reversed = ";".join(
                map(
                    str,
                    sorted(
                        [
                            seq_length - int(x) + 1
                            for x in (
                                pro.pro_ptm_locs if pro.pro_ptm_locs is not None else ""
                            ).split(";")
                            if x
                        ]
                    ),
                )
            )
            reversed_locs += [reversed]
        return reversed_locs

    def shuffle_pro_ptm_locs():
        # shuffle decoys are typically used for seqs that will not be further
        # proteolyzed; the one example we have is MHC which doesn't involve
        # any PTM.  Impl. this if needed, which will need the shuffle perm.
        if list(pros_df.pro_ptm_locs.unique()) != [""]:
            raise NotImplementedError("shuffle decoys don't support ptm locations")
        return pros_df.pro_ptm_locs

    decoy_ptm_transform = (
        reverse_pro_ptm_locs if decoy_mode == "reverse" else shuffle_pro_ptm_locs
    )
    # -------------------------------------------------------------------------

    decoys_df = pd.DataFrame(
        dict(
            pro_i=np.arange(n_pros, 2 * n_pros - 1),  # -1 to skip the nul
            pro_is_decoy=[True] * (n_pros - 1),  # same
            pro_id=[f"{decoy_mode[:3]}-{i}" for i in pros_df.pro_id],
            pro_ptm_locs=decoy_ptm_transform(),
        )
    )

    return decoys_df, decoy_seqs_df


def _step_4_proteolysis(pro_seqs_df, proteases):

    # TASK: Need ot fix this parallelization code...
    # pep_dfs = parallel_groupby_apply(pro_seqs_df.groupby("pro_i"), _proteolyze, compiled_protease_rules=compiled_protease_rules, _process_mode=True)
    # peps_df = pd.concat(pep_dfs)
    # peps_df = peps_df.reset_index(level=0).set_index(["pro_i", "pro_pep_i"])

    peps_df = (
        pro_seqs_df.groupby("pro_i")
        .apply(_proteolyze, proteases=proteases)
        .reset_index(level=0)
        .set_index(["pro_i", "pro_pep_i"])
    )

    # At this point the peps_df has a "local" index for the peptide -- ie. it restarts
    # every protein. But we want to concatenate all these into one big list of peptides.
    # The peps_df is indexed by "pro_i", "pro_pep_i" so a conversion table
    # is built "pro_pep_to_pep_i" and then this is merged with the peps_df causing
    # the "global" pep_i sequence to get into the pep_seqs_df and the "pro_pep_i"
    # can then be dropped.

    pro_pep_to_pep_i = peps_df.index.unique().to_frame().reset_index(drop=True)
    pro_pep_to_pep_i = pro_pep_to_pep_i.rename_axis("pep_i").reset_index()
    pep_seqs_df = pd.merge(
        left=pro_pep_to_pep_i, right=peps_df, on=["pro_pep_i", "pro_i"]
    ).drop(columns="pro_pep_i")

    # SET the pep_start and pep_stop based on the pep_offset_in_pro min and max of each pep_i group
    peps_df = pep_seqs_df.reset_index()[
        ["pro_i", "pep_i", "pep_offset_in_pro"]
    ].drop_duplicates()
    peps_df["pep_start"] = peps_df.groupby(["pep_i"]).pep_offset_in_pro.transform("min")
    peps_df["pep_stop"] = (
        peps_df.groupby(["pep_i"]).pep_offset_in_pro.transform("max") + 1
    )
    peps_df = (
        peps_df.drop(columns="pep_offset_in_pro")
        .drop_duplicates()
        .reset_index(drop=True)
    )[PrepResult.peps_columns]
    # [PrepResult.peps_columns] to reorder to canonical order avoiding warnings on concat etc.

    return peps_df, pep_seqs_df[["pep_i", "aa", "pep_offset_in_pro"]]


def _info(msg):
    """mock point"""
    info(msg)


def _do_ptm_permutations(df, n_ptms_limit):
    """
    df is a dataframe with a single pep_i, and pro_ptm_locs, and pep_offset_in_pro
    and aa for each location in the peptide
    """

    check.df_t(
        df,
        dict(pep_i=int, pro_ptm_locs=object, pep_offset_in_pro=int, aa=object),
        allow_extra_columns=True,
    )

    # pro_ptm_locs is identical for all rows
    pro_ptm_locs = df.pro_ptm_locs.values[0]
    if not pro_ptm_locs:
        return []

    # get 0-based indices from string representation; these are for the
    # entire protein.
    ptm_locs_zero_based = [(int(x) - 1) for x in pro_ptm_locs.split(";")]

    # get the ptms that coincide with the range spanned by this peptide.
    min_pos = df.pep_offset_in_pro.min()
    max_pos = df.pep_offset_in_pro.max()
    ptm_locs_zero_based = [x for x in ptm_locs_zero_based if min_pos <= x <= max_pos]

    n_ptms = len(ptm_locs_zero_based)
    if n_ptms > n_ptms_limit:
        _info(f"Skipping ptm for peptide {df.pep_i.iloc[0]} with {n_ptms} PTMs")

    if n_ptms_limit is not None and n_ptms > n_ptms_limit:
        return []

    powerset = [
        np.array(x)
        for length in range(1, len(ptm_locs_zero_based) + 1)
        for x in itertools.combinations(ptm_locs_zero_based, length)
    ]

    # powerset is a list of tuples.  So if have [2,4,10] in ptms you get
    # powerset = [ (2), (4), (10), (2,4), (2,10), (4,10), (2,4,10) ]
    #
    # The goal is to make a new peptide+seq for each of those tuples by
    # adding the modification '[p]' to the aa at that seq index location
    #

    mod = "[p]"

    new_pep_seqs = []

    for ptm_locs in powerset:
        new_pep_seq = df.copy()

        new_pep_seq.pep_i = np.nan

        new_pep_seq = new_pep_seq.set_index("pep_offset_in_pro")
        new_pep_seq.at[ptm_locs, "aa"] = new_pep_seq.aa[ptm_locs] + mod
        new_pep_seq = new_pep_seq.reset_index()

        new_pep_seqs += [new_pep_seq]

    return new_pep_seqs


def _step_5_create_ptm_peptides(peps_df, pep_seqs_df, pros_df, n_ptms_limit):
    """
    Create new peps and pep_seqs by applying PTMs based on the pro_ptm_locs information
    in pros_df.
    """

    # 1. Get subset of proteins+peps with ptms by filtering proteins with ptms and joining
    # to peps and pep_seqs
    #

    # This None vs "" is messy.

    pros_with_ptms = pros_df[pros_df.pro_ptm_locs != ""]
    df = (
        pros_with_ptms.set_index("pro_i").join(peps_df.set_index("pro_i")).reset_index()
    )
    df = df.set_index("pep_i").join(pep_seqs_df.set_index("pep_i")).reset_index()

    if len(df) == 0:
        return None, None

    # 2. for each peptide apply _do_ptm_permutations which will result in
    # a list of new dataframes of the form joined above; new_pep_infos is a
    # list of these lists.
    #
    # new_pep_infos = parallel_groupby_apply(
    #     df.groupby("pep_i"),
    #     _do_ptm_permutations,
    #     n_ptms_limit=n_ptms_limit,
    #     _trap_exceptions=False,
    #     _process_mode=True,
    # )
    new_pep_infos = zap.df_groups(
        _do_ptm_permutations,
        df.groupby("pep_i"),
        n_ptms_limit=n_ptms_limit,
        _trap_exceptions=False,
        _process_mode=True,
    )

    # 3. create new peps, pep_seqs, from list of dfs returned in (2)
    #
    #    peps_columns = ["pep_i", "pep_start", "pep_stop", "pro_i"]
    #    pep_seqs_columns = ["pep_i", "aa", "pep_offset_in_pro"]
    #
    new_peps = []
    new_pep_seqs = []
    pep_iz = peps_df.pep_i.unique()
    next_pep_i = peps_df.pep_i.max() + 1
    for new_peps_info in new_pep_infos:
        for pep_info in new_peps_info:
            # Note we only want one pep entry and pep_info contains enough rows to hold
            # the whole sequence for the peptide in the aa column.  So drop_duplicates()
            pep = pep_info[PrepResult.peps_columns].drop_duplicates()
            pep_seq = pep_info[
                PrepResult.pep_seqs_columns
            ].copy()  # avoid SettingWithCopyWarning with copy()

            pep.pep_i = next_pep_i
            pep_seq.pep_i = next_pep_i
            next_pep_i += 1

            new_peps += [pep]
            new_pep_seqs += [pep_seq]

    new_peps_df = pd.concat(new_peps)
    new_pep_seqs_df = pd.concat(new_pep_seqs)

    return new_peps_df, new_pep_seqs_df


def prep(prep_params, pro_spec_df):
    """
    Given protease and decoy mode, create proteins and peptides.

    Arguments:
        prep_params: PrepParams
        pro_spec_df: Columns: sequence (str), id (str), ptm_locs (str)

    Steps:
        1. Real proteins are checked for uniqueness in seq and id
        2. The real proteins are first string-split "unwound" into seq_ dataframes
           (one row per amino acid).
        3. The decoys are added by reversing those real DFs.
        4. The proteolysis occurs by a map against proteins
        5. PTMs are added

    ParamResults:
        Four DFs:
            * the pro data (one row per protein)
            * the pro_seq data (one row per aa) * n_pros
            * the pep data (one row per peptide)
            * the pep_seq data (one row per aa) * n_pres
    """

    if prep_params.drop_duplicates:
        pro_spec_df = pro_spec_df.drop_duplicates("sequence")
        pro_spec_df = pro_spec_df.drop_duplicates("name")

    _step_1_check_for_uniqueness(pro_spec_df)

    reals_df, real_seqs_df = _step_2_create_pros_and_pro_seqs_dfs(pro_spec_df)

    decoys_df, decoy_seqs_df = _step_3_generate_decoys(
        reals_df, real_seqs_df, prep_params.decoy_mode
    )

    pros_df = pd.concat((reals_df, decoys_df), sort=True).reset_index(drop=True)
    pros_df = pros_df.astype(dict(pro_i=int))

    pro_seqs_df = pd.concat((real_seqs_df, decoy_seqs_df)).reset_index(drop=True)

    peps_df, pep_seqs_df = _step_4_proteolysis(pro_seqs_df, prep_params.protease)

    if prep_params.n_peps_limit is not None:
        # This is used for debugging to limit the number of peptides.
        # This draws randomly to hopefully pick up decoys too
        n_peps = peps_df.pep_i.nunique()
        pep_iz = np.sort(
            np.random.choice(n_peps, prep_params.n_peps_limit, replace=False)
        )
        pep_iz[0] = 0  # Ensure the reserved value is present
        peps_df = peps_df.loc[pep_iz]
        pep_seqs_df = pep_seqs_df[pep_seqs_df.pep_i.isin(pep_iz)]

    if prep_params.n_ptms_limit != 0:
        # n_ptms_limit can be a non-zero value to limit the number of ptms
        # allowed per peptide, or set to 0 to skip ptm permutations even when
        # there are PTMs annotated for the proteins in protein_csv.
        ptm_peps_df, ptm_pep_seqs_df = _step_5_create_ptm_peptides(
            peps_df, pep_seqs_df, pros_df, prep_params.n_ptms_limit
        )
        if ptm_peps_df is not None and len(ptm_peps_df) > 0:
            peps_df = pd.concat([peps_df, ptm_peps_df])
            pep_seqs_df = pd.concat([pep_seqs_df, ptm_pep_seqs_df])
    # else:
    #     important("Skipping ptm permutations because n_ptms_limit is 0")

    return PrepResult(
        params=prep_params,
        _pros=pros_df,
        _pro_seqs=pro_seqs_df,
        _peps=peps_df,
        _pep_seqs=pep_seqs_df,
    )
