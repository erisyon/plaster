"""
A classifier for fluorosequences using with a Gaussian Mixture Model on top.

Terminology (applies to test_nn and test_nn)

    n_features:
        (n_cycles * n_channels)

    dye_count_space:
        The (n_cycles * n_channels) dimensional feature space of the data.

        Note that dye_count_space differs from "radiometry space" in that
        dye_count_space has unit brightness for each dye whereas radiometry_space
        has an independent brightness for each channel.

    dyemat:
        An instance of a matrix in dye_count_space

    dyerow:
        One row of a dyemat

    radiometry_space:
        The data space in which the data from the scope (or simulated) dwells.
        Like dye_count space this is an (n_cycles * n_channels) dimensional space,
        but wherein each channel has its own brightness and variance.

    radmat:
        An instance of a matrix in radiometry_space.
        Shape of n_rows * n_features

    radrow:
        One row of the radmat

    dye_gain:
        The brightness of each dye. Steps up-linearly with each additional dye-count.
        Note that this approximation may not hold for large numbers of dyes
        or different scopes/labels in that there is some evidence that single-molecule
        data should follow a log-normal distribution.

    variance_per_dye (VPD):
        The amount the variance goes up for each unit of dye. As noted above,
        there is some evidence that single molecule data should follow a log-normal
        distribution and therefore this model may be inadequate at higher dye counts.
        For now, this approximation seems sufficient.

    Gaussian Mixture Model (GMM):
        As the dye counts grows larger, the variance increases accordingly and thus
        individual fluoroseq classes tend to overlap at high dye_counts.
        The GMM is used to estimate the probability that a data-point came from
        a given fluoroseq.

    X:
        A matrix of n_rows by n_features holding sim or true signal data.

    uX:
        A matrix of X unit-normalized so that all channels are on a scale of 1 unit
        per dye.

    true_y:
    pred_y:
        A true or predicition vector of classification calls.


The NN classifier works by the following method:
    Calibration (off line):
        * Fits the ErrorModel (see error_model.py)

    Dye_tracks generated by monte-carlo simulations of peptides
    |
    |   Raw-data in radiometry-space from signal-processing or sim.
    |       |
    |       v
    +-> Find nearest-neighbor dye_track patterns within some neighbor-radius
            |
            v
        Weighted Gaussian Mixture Model to predict most likely dye_track in neighborhood.
            |
            v
        Maximum-Liklihood-Estimator to assign dye-track to peptide.


"""
import time
import numpy as np
import pandas as pd
from munch import Munch
from plaster.run.sim_v1.sim_v1_result import (
    ArrayResult,
    DyeType,
    IndexType,
    RadType,
    ScoreType,
)
from plaster.run.nn_v1.nn_v1_params import NNV1Params
from plaster.tools.schema import check
from plaster.tools.utils import data, utils
from plaster.tools.zap import zap
from plaster.vendor import pyflann
from scipy.spatial import distance


def _create_flann(dt_mat):
    pyflann.set_distance_type("euclidean")
    flann = pyflann.FLANN()
    flann.build_index(
        utils.mat_flatter(dt_mat).astype(RadType), algorithm="kdtree_simple"
    )
    return flann


def _get_neighbor_iz(flann, radrow, n_neighbors, radius, default=0):
    """
    Return n_neighbors worth of neighbors using radius search.
    If unable to find n_neighbors worth, pad the return array with default
    so that it will always return np.array((n_neighbors,))
    """
    check.array_t(radrow, ndim=1)

    # I don't think there's any guarantee that if you ask
    # for only one neighbor that you get the closest one.
    # Therefore, best not to use this under that assumption.
    # assert n_neighbors > 1
    # Removing this assert for now because my test results
    # show that getting 1 neighbor gives me nearly identical
    # results so I'm considering switching to a nearest-neighbor
    # model alone.

    nn_iz, dists = flann.nn_radius(radrow, radius, max_nn=n_neighbors)
    n_found = nn_iz.shape[0]
    neighbor_iz = np.full((n_neighbors,), default)
    neighbor_iz[0:n_found] = nn_iz[np.argsort(dists)]
    return neighbor_iz


def _do_nn(
    i: int,  # Row index to analyze
    nn_params: NNV1Params,
    radmat,  # Classify the [i] row of this (NOT normalized!)
    dt_mat,  # Targets
    dt_inv_var_mat,  # Inv variance of each target
    dt_weights,  # Weight of each target
    flann,  # Neighbor lookup index
    channel_i_to_gain_inv,  # Normalization term for each channel (radmat->unit_radmat)
    score_normalization,  # Normalization term to scale score by
    # dt_pep_sources_df,  # (dye_i, pep_i, n_rows). How often did pep_i express dye_i
    dye_to_best_pep_df,
    output_pred_dt_scores,
    output_pred_scores,
    output_pred_pep_iz,
    output_pred_dt_iz,
    output_true_dt_iz,
    true_dyemat=None,  # For debugging
):
    """
    Arguments:
        i: Index of the row
        radrow: One row of the radmat matrix -- not this is NOT normalized!
        dyerow: The true dyerow used for debugging
        dt_mat: The Dyetrack object used for neighbor lookup.
        dt_inv_var_mat: The inverse variance of each pattern (based on the VPD for each count)
        dt_weights: The weight of the pattern
        flann: The neighbor lookup
        The remainder are temporary parameters until I find the best solution
    """
    radrow = radmat[i]
    unit_radrow = radrow * channel_i_to_gain_inv[:, None]

    check.array_t(dt_mat, ndim=3)
    n_dts, n_channels, n_cycles = dt_mat.shape
    check.array_t(dt_inv_var_mat, shape=dt_mat.shape)
    check.array_t(radrow, shape=(n_channels, n_cycles))
    check.array_t(dt_weights, shape=(n_dts,))

    n_cols = n_channels * n_cycles
    unit_radrow_flat = unit_radrow.reshape((n_cols,))
    if true_dyemat is not None:
        dyerow = true_dyemat[i].reshape((n_cols,))
        # If true_dyemat is available then we can "cheat" and find the
        # true_dt_i for debugging purposes only.
        # Note, there is no guarantee that the training set has every row
        # that is represented by the test set. Thus the "true_dt_i"
        # might not be found. We find it by using the _get_neighbor_iz
        # with a tiny radius because if the test dyetrack is found
        # among the targets it should be right on top of its target.
        output_true_dt_iz[i] = _get_neighbor_iz(
            flann,
            radrow=dyerow.astype(RadType),
            n_neighbors=nn_params.n_neighbors,
            radius=0.01,
            default=0,
        )[0]

    # # FIND the neighbors in dyetrack target space encoded in the FLANN index
    nn_iz = _get_neighbor_iz(
        flann,
        unit_radrow_flat,
        n_neighbors=nn_params.n_neighbors,
        radius=nn_params.radius,
        default=-1,
    )

    # REMOVE -1 = unfound then convert to unsigned
    nn_iz = nn_iz[nn_iz >= 0].astype(IndexType)

    # FILTER any low-weight dyetracks
    sufficient_weight_mask = dt_weights[nn_iz] >= nn_params.dt_filter_threshold
    nn_iz = nn_iz[sufficient_weight_mask]
    n_neigh_found = np.sum(nn_iz >= 0)

    composite_score = ScoreType(0.0)
    pred_pep_i = IndexType(0)
    pred_dt_i = IndexType(0)
    pred_dt_score = ScoreType(0.0)

    if n_neigh_found > 0:
        assert 1 <= n_neigh_found <= nn_params.n_neighbors
        neigh_dt_mat = dt_mat[nn_iz].reshape((n_neigh_found, n_cols))
        neigh_dt_inv_var = dt_inv_var_mat[nn_iz].reshape((n_neigh_found, n_cols))
        neigh_weights = dt_weights[nn_iz]

        def cd():
            return distance.cdist(
                unit_radrow_flat.reshape((1, n_cols)),
                neigh_dt_mat,
                metric=nn_params.dt_score_metric,
            )[0]

        def penalty():
            p = 1.0
            if nn_params.rare_penalty is not None:
                p *= 1.0 - np.exp(-nn_params.rare_penalty * neigh_weights)

            if nn_params.penalty_coefs is not None:
                # Experimental: reduce score by total est. dye count
                total_brightness = unit_radrow.sum()
                # From fitting on RF I see that p_correct is correlated to
                # total brightness. A linear function m=0.054, b=0.216
                # So I'm going to try reducing score by this factor
                correction = max(
                    0.0,
                    min(
                        1.0,
                        (
                            total_brightness * nn_params.penalty_coefs[0]
                            + nn_params.penalty_coefs[1]
                        ),
                    ),
                )
                assert 0 <= correction <= 1.0
                p *= correction

            return p

        # if nn_params.dt_score_mode == "gmm_normalized_wpdf":
        #     delta = unit_radrow_flat - neigh_dt_mat
        #     vdist = np.sum(delta * delta * neigh_dt_inv_var, axis=1)
        #     pdf = np.exp(-vdist)
        #     weighted_pdf = neigh_weights * pdf
        #     scores = utils.np_safe_divide(weighted_pdf, weighted_pdf.sum())

        if nn_params.dt_score_mode == "gmm_normalized_wpdf_dist_sigma":
            """
            https://en.wikipedia.org/wiki/Multivariate_normal_distribution
            Given:
                Sigma: covariance matrix
                mu: mean
                k: Dimensionality of the Gaussian
            The multivariate normal has the form:
                (
                    (2.0 * np.pi)**(-k / 2.0)
                ) * (
                    np.linalg.det(Sigma)**(-1.0 / 2.0)
                ) * np.exp(
                    -1.0 / 2.0 * ((x-mu).T @ np.linalg.inv(Sigma) @ (x-mu))
                )

            We can make some simplifications here:

            1.  We have n_rows (number of neighbors) and n_cols (number of feature dimensions)

            2.  We have pre-computed (x-mu) and call it "delta"
                This is a (n_rows, n_cols) matrix

            3.  We don't actually have Sigma, rather we have
                the "inverse variance" which we call: "neigh_dt_inv_var"
                and which we store in vector form!
                Therefore:
                   Sigma = 1.0 / neigh_dt_inv_var
                This is a (n_rows, n_cols) matrix

            4.  Our covariance matrix (Sigma) is diagonal and therefore
                its determinant is the product of the diagonal elements
                which, again, is stored as the rows.
                   np.linalg.det(Sigma) == np.prod(Sigma, axis=1)

            5.  Following from Sigma being diagonal, its inverse is:
                (1.0 / its elements). Therefore:
                   np.linalg.inv(Sigma) == 1.0 / Sigma

            6.  Furthermore, the term:
                   (x-mu).T @ np.linalg.inv(Sigma) @ (x-mu)
                is typically a full matrix expression.
                But Sigma is diagonal and stored as a vector, therefore:
                    delta.T @ np.sum((1.0 / Sigma) * delta, axis=1) ==
                    np.sum(delta * ((1.0 / Sigma) * delta), axis=1)

            7.  Because the whole equation converts to vector form
                all of the row operations on each neighbor can
                be done simultaneously.

            8.  The  (2.0 * np.pi)**(-k / 2.0) is omitted as it will get
                factored out when scores are computed.

            Therefore:
                n_rows = n_neigh_found = # Number of rows of neigh_dt_mat
                n_cols = # Number of columns of neigh_dt_mat
                delta = unit_radrow_flat - neigh_dt_mat  # np.array(size=(n_rows, n_cols,))
                Sigma = 1.0 / neigh_dt_inv_var  # np.array(size=(n_rows, n_cols,))

                pdf_at_x = (
                (2.0 * np.pi)**(-n_cols / 2.0)  # A constant (we can factor this out)
                ) * (
                np.prod(Sigma, axis=1)**(-1.0 / 2.0)  # np.array(size=(n_rows,))
                ) * np.exp(
                -1.0 / 2.0 * np.sum(delta * (neigh_dt_inv_var * delta), axis=1)
                )  # np.array(size=(n_rows,))
            """
            delta = unit_radrow_flat - neigh_dt_mat
            vdist = np.sum(delta * neigh_dt_inv_var * delta, axis=1)
            sigma = 1.0 / neigh_dt_inv_var
            determinant_of_sigma = np.prod(sigma, axis=1)
            pdf = determinant_of_sigma ** (-1 / 2) * np.exp(-vdist / 2)
            weighted_pdf = neigh_weights * pdf
            scores = utils.np_safe_divide(weighted_pdf, weighted_pdf.sum())

        # elif nn_params.dt_score_mode == "gmm_normalized_wpdf_no_inv_var":
        #     delta = unit_radrow_flat - neigh_dt_mat
        #     vdist = np.sum(delta * delta, axis=1)
        #     pdf = np.exp(-vdist)
        #     weighted_pdf = neigh_weights * pdf
        #     scores = utils.np_safe_divide(weighted_pdf, weighted_pdf.sum())
        #
        # elif nn_params.dt_score_mode == "cdist_normalized":
        #     d = cd()
        #     scores = 1.0 / (nn_params.dt_score_bias + d)
        #     scores = utils.np_safe_divide(scores, scores.sum())
        #
        # elif nn_params.dt_score_mode == "cdist_weighted_sqrt":
        #     d = cd()
        #     scores = np.sqrt(neigh_weights) / (nn_params.dt_score_bias + d)
        #
        # elif nn_params.dt_score_mode == "cdist_weighted_log":
        #     d = cd()
        #     scores = np.log(neigh_weights) / (nn_params.dt_score_bias + d)
        #
        # elif nn_params.dt_score_mode == "cdist_weighted_normalized":
        #     d = cd()
        #     scores = neigh_weights / (nn_params.dt_score_bias + d)
        #     scores = utils.np_safe_divide(scores, scores.sum())
        #
        # elif nn_params.dt_score_mode == "cdist_weighted_normalized_sqrt":
        #     d = cd()
        #     scores = np.sqrt(neigh_weights) / (nn_params.dt_score_bias + d)
        #     scores = utils.np_safe_divide(scores, scores.sum())
        #
        # elif nn_params.dt_score_mode == "cdist_weighted_normalized_log":
        #     d = cd()
        #     scores = np.log(neigh_weights) / (nn_params.dt_score_bias + d)
        #     scores = utils.np_safe_divide(scores, scores.sum())

        else:
            raise NotImplementedError()

        # PICK highest score
        scores *= penalty()
        scores = scores.astype(ScoreType)
        arg_sort = np.argsort(scores)[::-1]
        best_arg = arg_sort[0].astype(int)
        pred_dt_i = nn_iz[best_arg]
        pred_dt_score = scores[best_arg]

        assert type(pred_dt_i) is IndexType

        assert 0 <= pred_dt_score <= 1.0

        # At this point we have a dyetrack prediction and
        # we do a simple-minded call of the most likely
        # peptide for that dyetrack and the composite

        # But I have a problem -- before I was normalizing the dt_scores
        # before I multiplied by the pep prob.
        # But I think that's still okay.

        # USE the dt prediction to find a peptide prediction
        # df = dt_pep_sources_df
        # dt_pep_mat = df[df.dye_i == pred_dt_i].values
        # # dt_pep_mat has three columns:
        # #     0:dye_i (a constance == pred_dt_i because of previ lin filter)
        # #     1:pep_i (the peptide this dyetrack camer from)
        # #     2:n_rows (number of times that pep_i generated this dyetrack)
        #
        # sorted_pep_mat_args = np.argsort(dt_pep_mat[:, 2])[::-1]
        # top_pep_row_i = sorted_pep_mat_args[0]
        # pred_pep_i = dt_pep_mat[top_pep_row_i, 1]
        # pred_pep_score = dt_pep_mat[top_pep_row_i, 2] / dt_pep_mat[:, 2].sum()

        best_pep_row = dye_to_best_pep_df.loc[pred_dt_i]
        pred_pep_i = best_pep_row.pep_i
        composite_score = min(
            1.0, (pred_dt_score * best_pep_row.score) / score_normalization
        )
        assert 0 <= composite_score <= 1.0

    output_pred_dt_scores[i] = ScoreType(pred_dt_score)
    output_pred_scores[i] = ScoreType(composite_score)
    output_pred_pep_iz[i] = pred_pep_i.astype(IndexType)
    output_pred_dt_iz[i] = pred_dt_i.astype(IndexType)


'''
def _fit_gain_one_channel(one_channel_radmat, expected_dark_cycle):
    """
    Fit the gain of one_channel_radmat

    Assumes (demands) that the dye count is not more than one.
    This will fail in all other cases.

    Arguments:
        one_channel_radmat: A 2D matrix of (n_samples, n_cycles)
        expected_dark_cycle: cycle (0-based) where the dark is expected

    Returns:
        Gain estimate
    """
    from sklearn.cluster import KMeans  # Defer slow import

    check.array_t(one_channel_radmat, ndim=2)
    n_rows, n_cycles = one_channel_radmat.shape
    assert np.any(one_channel_radmat > 100.0)  # Check that this is a non-unity X

    n_rows = one_channel_radmat.shape[0]

    # Step one, divide the measurements into two groups (dark and bright) by using k-means

    # sklearn's KMeans only accept 2+ dimensional data, so convert it
    samples = one_channel_radmat.flatten()
    samples_2d = np.zeros((samples.shape[0], 2))
    samples_2d[:, 0] = samples
    kmeans = KMeans(n_clusters=2, random_state=0).fit(samples_2d)
    gain = np.median(samples[kmeans.labels_ == 1])
    dark = np.sort(samples[kmeans.labels_ == 1])[0]
    # gain is now an initial guess at the one-dye gain and
    # dark is a lower bound

    # Step 2: Filter outliers and refine estimate

    # Filter out any rows that don't conform to the expected pattern
    # by measuring the distance of each row in pattern space to the expected
    expected_pattern = np.ones((n_cycles,))
    expected_pattern[expected_dark_cycle:] = 0

    keep_mat = None
    for i in range(5):  # 5 is empirical
        # Repeat solving for the gain keeping anything that is < 2 stdev from the gain
        # This knocks out high corruptions
        delta = one_channel_radmat / gain - expected_pattern
        dist = np.sqrt(np.sum(delta * delta, axis=1))

        keep_mask = dist < 1.0
        keep_mat = one_channel_radmat[keep_mask]

        # Take samples from rows that match the pattern and that aren't dark
        samples = keep_mat.flatten()
        samples = samples[samples > dark]

        # Keep samples that are 2 std from current gain
        std = np.std(samples - gain)
        samples = samples[(samples - gain) ** 2 < (2 * std) ** 2]

        gain = np.mean(samples)

    return gain, keep_mat


def _fit_vpd_one_channel(one_channel_radmat, gain, expected_dyerow, accept_dist=1.0):
    n_rows, n_cycles = one_channel_radmat.shape
    assert np.any(one_channel_radmat > 100.0)  # Check that this is a non-unity X
    assert expected_dyerow.shape[0] == n_cycles
    n_dyes = int(np.max(expected_dyerow))

    radmat = one_channel_radmat / gain

    # Filter out any rows that don't conform to the expected pattern
    # by measuring the distance of each row in pattern space to the expected
    delta = radmat - expected_dyerow
    #     debug(delta[:, 0:])
    #     debug(expected_dyerow)
    delta = delta * np.exp(-0.2 * expected_dyerow)
    #    delta[:, expected_dyerow == 0.0] = 0.0
    #     debug(delta[:, 0:])
    dist = np.sqrt(np.sum(delta * delta, axis=1))

    # Take samples from rows that match the pattern
    keep_rows = dist < accept_dist
    keep_radmat = radmat[keep_rows]

    x = [0]
    y = [0]
    for i in range(1, n_dyes + 1):
        cycles_with_this_many_dyes = np.argwhere(expected_dyerow == i).flatten()
        samples = keep_radmat[:, cycles_with_this_many_dyes]
        x += [i]
        y += [np.var(samples)]

    x = np.array(x)
    y = np.array(y)
    m, b = np.polyfit(x, y, 1)

    return m, b, n_dyes, x, y, keep_radmat
'''


def _cpu_count():
    """mock-point"""
    from multiprocessing import cpu_count

    return cpu_count()


def _do_batch_unique(rng, dyemat):
    return np.unique(dyemat[rng], return_inverse=True, return_counts=True, axis=0)


def _step_1_create_neighbors_lookup_singleprocess(true_pep_iz, dyemat, output_dt_mat):
    """
    The dyemat may have many duplicate rows, each from some number of peps.

    These duplicate rows are consolidated so that each coordinate in dyemat space
    is given a unique "dye_i".

    The unique (sorted) dyetracks are written to output_dt_mat which is expected
    to be large enough to hold them.

    Returns:
        dyetracks_df: DF(dye_i, weight).
            Where weight is the sum of all rows that pointed to this dyetrack
        dt_pep_sources_df: DF(dye_i, pep_i, n_rows)
            Records how many times each peptide generated dye_i where count > 0.
        flann: A fast Approximate Nearest Neighbors lookup using PYFLANN.
        n_dts: Number of actual unique dts
    """
    check.array_t(dyemat, ndim=3)  # (n_peps * n_samples, n_channels, n_cycles): uint8

    n_rows, n_channels, n_cycles = dyemat.shape

    # Example usage of unique
    # b = np.array([1, 4, 3, 2, 1, 2])
    # p = np.unique(b, return_inverse=True, return_counts=True, )
    # p == (array([1, 2, 3, 4]), array([0, 3, 2, 1, 0, 1]), array([2, 2, 1, 1]))
    dt_mat, true_dt_iz, dt_counts = np.unique(
        dyemat, return_inverse=True, return_counts=True, axis=0
    )

    n_dts = dt_mat.shape[0]
    output_dt_mat[0:n_dts] = dt_mat

    # Check that the nul row exists and it the first element
    if not np.all(dt_mat[0] == 0):
        raise ValueError("No null row was included in the dyemat")

    flann = _create_flann(dt_mat)

    dyetracks_df, dt_pep_sources_df, dye_to_best_pep_df = _setup_pep_source_dfs(
        true_dt_iz, true_pep_iz, dt_counts
    )
    return dyetracks_df, dt_pep_sources_df, dye_to_best_pep_df, flann, n_dts


def _setup_pep_source_dfs(true_dt_iz, true_pep_iz, dt_counts):
    dyetracks_df = (
        pd.DataFrame(dict(weight=dt_counts))
        .reset_index()
        .rename(columns=dict(index="dye_i"))
    )

    dt_pep_sources_df = (
        pd.DataFrame(dict(dye_i=true_dt_iz, pep_i=true_pep_iz))
        .groupby(["dye_i", "pep_i"])
        .size()
        .to_frame("n_rows")
        .reset_index()
    )

    # BUILD a lookup for the most likely peptide per dyetrack and its score
    df_most_likely_pep_per_dye = (
        dt_pep_sources_df.sort_values("n_rows", ascending=False)
        .drop_duplicates(["dye_i"])
        .set_index("dye_i")
    )
    df_row_totals_per_dye = (
        dt_pep_sources_df.groupby("dye_i")[["n_rows"]]
        .sum()
        .rename(columns=dict(n_rows="total"))
    )
    dye_to_best_pep_df = df_most_likely_pep_per_dye.join(df_row_totals_per_dye)
    dye_to_best_pep_df["score"] = dye_to_best_pep_df.n_rows / dye_to_best_pep_df.total

    return dyetracks_df, dt_pep_sources_df, dye_to_best_pep_df


def _step_2_create_inverse_variances(dt_mat, channel_i_to_vpd):
    """
    Using the Variance Per Dye find the inverse for each row of dyemat.
    This deals with zero-dyes by assigning the half the variance of the 1-count dye.
    vpd stands for variance per dye. Our models indicate that the standard deviation
    goes up roughly linearly with the number of dyes.
    The later code (_do_nn_and_gmm) needs the inverse variance, so we square the standard deviation to obtain the
    variance and take the inverse.
    Arguments:
        dt_mat is the unique dyetracks

    Returns:
        ndarray(n_rows, n_channels * n_cycles): inverse variance for each row (flatten)
    """

    check.array_t(dt_mat, ndim=3)
    check.array_t(channel_i_to_vpd, ndim=1)
    # Variances of zero will cause div by zeros so all zeros
    # are set to 0.5 which is chosen arbitrarily because it is > 0 and < 1.
    dt_mat = dt_mat.astype(float)
    dt_mat[dt_mat == 0] = 0.5
    vpd_broadcast = channel_i_to_vpd[None, :, None]
    spd = np.sqrt(vpd_broadcast)
    return 1.0 / np.square(
        spd * dt_mat
    )  # Scaling by the standard deviation per dye by channel


def nn(nn_params, sim_result, radmat, true_dyemat=None, progress=None):
    """
    Main entrypoint for nearest_neighbors.

    Arguments:
        nn_params: TestNNParams
        sim_result: SimV1Result -- Uses the train_* values
        radmat: The radmat to classify.
        true_dyemat: Optional for debugging -- the dyemat of the radmat
            ie. the dyerow that corresponds to each radrow.
        progress: Optional progress callback

    Returns:
        pred_pep_iz
        scores

    This is composed of the following steps:
        1. Create a unit radmat
        2. Create a unique dyetrack mat (dt_mat); these are the
           "neighbors" that will be searched.
        3. Create inverse variance for each row of dt_mat; inv_var_dt_mat
        4. Classify each row of the unit radmat with the Gaussian Mixture Model.
    """

    # Allocate the dt_mat as large as it COULD possibly be
    # and then after populating it with the unique values
    # we can resize if using dt_mat.base.resize(n_bytes)
    # The max size is the (extremely unlikely) value of
    # n_peps * n_samples
    check.array_t(radmat, ndim=3, dtype=RadType)
    check.array_t(sim_result.train_dyemat, ndim=3, dtype=DyeType)
    shape = sim_result.train_dyemat.shape
    n_dts_max = shape[0]
    n_channels, n_cycles = shape[1:]
    dt_mat = ArrayResult(
        "dt_mat", DyeType, shape=(n_dts_max, n_channels, n_cycles), mode="w+"
    )

    (
        dyetracks_df,
        dt_pep_sources_df,
        dye_to_best_pep_df,
        flann,
        n_dts,
    ) = _step_1_create_neighbors_lookup_singleprocess(
        sim_result.train_true_pep_iz, sim_result.train_dyemat, output_dt_mat=dt_mat.arr(),
    )

    # dyetracks_df: (dye_i, weight)
    # dt_pep_sources_df: (dye_i, pep_i, n_rows)
    assert n_dts <= n_dts_max and n_dts == dyetracks_df.dye_i.max() + 1

    # Collapse the dt_mat to the actual number of rows.
    # This will cause the memmap file to truncate in size.
    dt_mat.reshape((n_dts, n_channels, n_cycles))

    # dt_mat is the dyetrack mat of the TARGETS as build by the training set
    # Not to be confused with dyemat which is the dyemat of the test points
    # There is no guarantee that the dyerow of a test point is even *in*
    # the training set.

    dt_inv_var_mat = _step_2_create_inverse_variances(
        dt_mat.arr(), np.array(sim_result.params.channel_i_to_vpd)
    )

    dt_weights = dyetracks_df.reindex(np.arange(n_dts), fill_value=0).weight.values

    channel_i_to_gain_inv = (
        1.0 / np.array(sim_result.params.channel_i_to_gain)
    ).astype(RadType)

    # Now classify each radrow
    check.array_t(radmat, ndim=3)
    n_rows = radmat.shape[0]
    if true_dyemat is not None:
        assert true_dyemat.shape == radmat.shape

    pred_dt_scores = ArrayResult("pred_dt_scores", ScoreType, (n_rows,), mode="w+")
    pred_scores = ArrayResult("pred_scores", ScoreType, (n_rows,), mode="w+")
    pred_pep_iz = ArrayResult("pred_pep_iz", IndexType, (n_rows,), mode="w+")
    pred_dt_iz = ArrayResult("pred_dt_iz", IndexType, (n_rows,), mode="w+")
    true_dt_iz = ArrayResult("true_dt_iz", IndexType, (n_rows,), mode="w+")

    # Score normalization requires knowing about the distribution of
    # scores but I do not want to make two full passes over the dataset.
    # To avoid this, I randomly sample a fraction of the dataset
    # to collect the score distribution and then I pass in a normalization
    # term into the second pass.

    if nn_params.random_seed is None:
        nn_params.random_seed = int(time.time())
    # prof()

    zap.arrays(
        _do_nn,
        dict(i=np.arange(n_rows)),
        nn_params=nn_params,
        radmat=radmat,
        dt_mat=dt_mat.arr(),
        dt_inv_var_mat=dt_inv_var_mat,
        dt_weights=dt_weights,
        flann=flann,
        channel_i_to_gain_inv=channel_i_to_gain_inv,
        score_normalization=1.0,
        dye_to_best_pep_df=dye_to_best_pep_df,
        output_pred_dt_scores=pred_dt_scores.arr(),
        output_pred_scores=pred_scores.arr(),
        output_pred_pep_iz=pred_pep_iz.arr(),
        output_pred_dt_iz=pred_dt_iz.arr(),
        output_true_dt_iz=true_dt_iz.arr(),
        true_dyemat=true_dyemat,
        _progress=progress,
    )

    return Munch(
        dt_mat=dt_mat,
        dyetracks_df=dyetracks_df,
        dt_pep_sources_df=dt_pep_sources_df,
        true_dt_iz=true_dt_iz,
        pred_dt_iz=pred_dt_iz,
        dt_scores=pred_dt_scores,
        scores=pred_scores,
        pred_pep_iz=pred_pep_iz,
    )
