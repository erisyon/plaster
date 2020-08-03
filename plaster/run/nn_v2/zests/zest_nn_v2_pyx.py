import numpy as np
from plaster.run.nn_v2.fast import nn_v2_fast


def zest_fast_nn():
    train_dyemat = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0], [2, 1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 2, 1, 0, 0],],
        dtype=np.uint8,
    )

    train_dyepeps = np.array(
        [
            [0, 0, 10],  # Dye track 0 comes from pep 0 10 times
            [1, 1, 10],  # Dye track 1 comes from pep 1 10 times
            [1, 2, 5],  # Dye track 1 comes from pep 2 5 times
            [2, 2, 5],  # Dye track 2 comes from pep 2 5 times
        ],
        dtype=np.uint64,
    )

    test_unit_radmat = np.array(
        [
            [1.1, 0.9, 0.0, 0.0, 1.9, 1.1, 0.0, 0.1],  # from dye track 2
            [2.1, 0.9, 1.1, 1.0, 1.1, 0.0, 0.1, 0.0],  # from dye track 1
        ],
        dtype=np.float32,
    )

    pred_pep_iz, scores, pred_dye_iz = nn_v2_fast.fast_nn(
        test_unit_radmat, train_dyemat, train_dyepeps, n_neighbors=2
    )
    assert pred_pep_iz.tolist() == [2, 1]

    # TODO: Assert on scores?
    # debug(pred_pep_iz, scores)
