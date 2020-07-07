from munch import Munch
from plaster.tools.schema.schema import Schema as s, Params


class TestNNParams(Params):
    defaults = Munch(
        include_training_set=False,
        n_neighbors=8,
        dt_score_mode="gmm_normalized_wpdf_dist_sigma",
        dt_score_metric="",
        dt_score_bias=0.1,
        dt_filter_threshold=0,
        rare_penalty=0.8,
        penalty_coefs=None,
        radius=15.0,
        random_seed=None,
    )

    schema = s(
        s.is_kws_r(
            include_training_set=s.is_bool(),
            n_neighbors=s.is_int(),
            dt_score_bias=s.is_float(),
            dt_score_mode=s.is_str(
                options=[
                    "gmm_normalized_wpdf",
                    "gmm_normalized_wpdf_dist_sigma",
                    "gmm_normalized_wpdf_no_inv_var",
                    "one",
                    "dt_freq_log_weight",
                    "cdist_normalized",
                    "cdist_weighted_sqrt",
                    "cdist_weighted_log",
                    "cdist_weighted_normalized",
                    "cdist_weighted_normalized_sqrt",
                    "cdist_weighted_normalized_log",
                ]
            ),
            dt_score_metric=s.is_str(
                options=[
                    "",
                    "braycurtis",
                    "canberra",
                    "chebyshev",
                    "cityblock",
                    "correlation",
                    "cosine",
                    "euclidean",
                    "jensenshannon",
                    "minkowski",
                    "seuclidean",
                    "sqeuclidean",
                ]
            ),
            dt_filter_threshold=s.is_int(),
            penalty_coefs=s.is_list(
                elems=s.is_float(), min_len=2, max_len=2, noneable=True
            ),
            rare_penalty=s.is_float(noneable=True),
            radius=s.is_float(),
            random_seed=s.is_int(noneable=True),
        )
    )
