import numpy as np
import pandas as pd
from plaster.run.survey_nn_v1.survey_nn_v1_result import SurveyNNV1Result
from plaster.tools.aaseq.aaseq import aa_str_to_list
from plaster.tools.schema import check
from scipy.spatial.distance import cdist


def _euc_dist(sim_result):
    """
    Computes euclidean distance between all dye-tracks produced by a simulation.
    Returns three parallel arrays:
        pep_iz: pep indices
        nn_pep_iz: pep indices corresponding to nearest neighbor of pep_i
        nn_dist: distance from pep_i it's nearest neighbor nn_pep_i

    Notes: here we always compute all-vs-all distances.  But if there are
    proteins-of-interest, we could compute only the distances from those
    to the set of all, and have a much-reduced problem-size.  At the moment,
    this protein-of-interest filter is applied during the reports.
    """

    check.array_t(sim_result.train_dyemat, ndim=3)
    shape = sim_result.train_dyemat.shape
    dyemat = sim_result.train_dyemat.reshape((shape[0], shape[1] * shape[2]))
    d = cdist(dyemat, dyemat)
    np.fill_diagonal(d, np.nan)
    nn_dist_args = np.nanargmin(d, axis=1)

    # the dim of train_dyemat above does not include "dark" peptides,
    # so we'll need to factor that in to create a table for all peps.
    n_peps = len(sim_result.train_pep_recalls)
    # dark_pep_mask = sim_result.train_pep_recalls == 0.0

    # we'll store nearest neighbor pep_i and distance
    neighbor_pep_i = np.full([n_peps,], -1)
    neighbor_dist = np.full([n_peps,], -1)

    neighbor_pep_i = sim_result.train_true_pep_iz[nn_dist_args]
    neighbor_dist = d[np.arange(d.shape[0]), nn_dist_args]

    return range(n_peps), neighbor_pep_i, neighbor_dist


def survey_nn_v1(
    survey_nn_params, prep_result, sim_result, progress=None, pipeline=None
):
    """
    Compute a distance between between peptides that exist in prep_result
    using the dye-tracks employed by nearest-neighbor.  Create a DF that
    collects these distances with other information useful in surveying
    a number of protease-label schemes to determine which ones are well
    suited to some informatics objective, such as identifying a protein(s).

    Notes:
        - We are not including decoys.  If you want to include decoys (assuming they
          were used in the simulation) use the test dyemat rather than train.

    """

    # get simple euclidean nearest-neighbor info & store in Dataframe
    pep_iz, nn_pep_iz, nn_dist = _euc_dist(sim_result)
    df = pd.DataFrame()
    df["pep_i"] = pep_iz
    df["nn_pep_i"] = nn_pep_iz
    df["nn_dist"] = nn_dist

    # Join this to some flu information so we have it all in one place, especially
    # info about degeneracy (when more than one pep has the same dyetrack)
    # This isn't very DRY, since this data already lives in the prep and sim results.
    # But it makes downstream report-code simpler and faster to filter and search
    # these results if everything you need is already joined in one DF.
    # My approach is to put everything into the SurveyResult that you want
    # to be able to filter on to minimize computation in the report.
    # This is possible for nearly everything, except things you want to
    # be able to change at report time, like what PTMs you're interested in
    # if this survey involves PTMs.
    #
    peps__flus = sim_result.peps__flus(prep_result)
    peps__flus["pep_len"] = peps__flus.apply(
        lambda x: x.pep_stop - x.pep_start - 1, axis=1
    )

    # include the peptide sequence, and whether it has Proline at position 2
    pepstrs = prep_result.pepstrs()
    pepstrs["P2"] = pepstrs.apply(
        lambda row: True
        if row.seqstr and len(row.seqstr) > 1 and aa_str_to_list(row.seqstr)[1] == "P"
        else False,
        axis=1,
    )

    df = (
        df.set_index("pep_i")
        .join(peps__flus.set_index("pep_i"), how="left")
        .join(pepstrs.set_index("pep_i"), how="left")
        .reset_index()
    )[SurveyNNV1Result.survey_columns]

    return SurveyNNV1Result(params=survey_nn_params, _survey=df)
