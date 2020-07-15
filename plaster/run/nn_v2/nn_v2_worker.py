from plaster.run.nn_v2.fast import nn_v2_fast


def nn_worker(nn_v2_params, sim_v2_result):

    train_dyemat = sim_v2_result.train_dyemat
    shape = train_dyemat.shape
    train_dyemat = train_dyemat.reshape((shape[0] * shape[1], shape[2] * shape[3]))

    pred_iz, scores = nn_v2_fast.fast_nn(
        test_radmat, train_dyemat, train_dyepeps, n_neighbors=2
    )
