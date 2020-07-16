from plaster.run.nn_v2.nn_v2_result import NNV2Result
from plaster.run.nn_v2.fast import nn_v2_fast


def nn_v2_worker(nn_v2_params, sim_v2_result):
    import pudb; pudb.set_trace()

    test_pred_pep_iz, test_scores = nn_v2_fast.fast_nn(
        sim_v2_result.flat_test_radmat(),
        sim_v2_result.flat_train_dyemat(),
        sim_v2_result.train_dyepeps,
        n_neighbors=nn_v2_params.n_neighbors
    )

    return NNV2Result(
        test_pred_pep_iz=test_pred_pep_iz,
        test_scores = test_scores,
    )
