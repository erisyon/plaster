import numpy as np
from plaster.tools.schema import check
from plaster.run.base_result import ArrayResult
from plaster.run.call_bag import CallBag
from plaster.run.test_nn.test_nn_result import TestNNResult
from plaster.run.test_nn.nn import nn
from plaster.run.sim_v1.sim_v1_result import IndexType
from plaster.tools.utils import utils
from plaster.tools.log.log import debug, prof


def test_nn(test_nn_params, prep_result, sim_result, progress=None, pipeline=None):
    n_channels, n_cycles = sim_result.params.n_channels_and_cycles

    n_phases = 6 if test_nn_params.include_training_set else 3
    if pipeline is not None:
        pipeline.set_phase(0, n_phases)

    shape = sim_result.test_radmat.shape
    assert len(shape) == 4
    test_radmat = sim_result.test_radmat.reshape(
        (shape[0] * shape[1], shape[2], shape[3])
    )
    test_dyemat = sim_result.test_dyemat.reshape(
        (shape[0] * shape[1], shape[2], shape[3])
    )
    test_result = nn(
        test_nn_params,
        sim_result,
        radmat=test_radmat,
        true_dyemat=test_dyemat,
        progress=progress,
    )

    test_result.true_pep_iz = ArrayResult(
        filename="test_true_pep_iz",
        shape=(shape[0] * shape[1],),
        dtype=IndexType,
        mode="w+",
    )
    test_result.true_pep_iz[:] = np.repeat(
        np.arange(shape[0]).astype(IndexType), shape[1]
    )
    check.t(test_result.true_pep_iz, ArrayResult)
    check.t(test_result.pred_pep_iz, ArrayResult)

    call_bag = CallBag(
        true_pep_iz=test_result.true_pep_iz.arr(),
        pred_pep_iz=test_result.pred_pep_iz.arr(),
        scores=test_result.scores.arr(),
        prep_result=prep_result,
        sim_result=sim_result,
    )

    if pipeline is not None:
        pipeline.set_phase(1, n_phases)

    test_result.peps_pr = call_bag.pr_curve_by_pep(progress=progress)

    # If there is abundance information, compute the abundance-adjusted PR
    # This call returns None if there is no abundance info avail.
    if pipeline is not None:
        pipeline.set_phase(2, n_phases)

    test_result.peps_pr_abund = call_bag.pr_curve_by_pep_with_abundance(
        progress=progress
    )

    if test_nn_params.include_training_set:
        # Permit testing for over-fitting by classifying on the train data

        if pipeline is not None:
            pipeline.set_phase(3, n_phases)

        real_pep_iz = prep_result.peps__no_decoys().pep_i.values
        keep_rows = np.isin(sim_result.train_true_pep_iz, real_pep_iz)
        train_radmat = sim_result.train_radmat[keep_rows]
        train_dyemat = sim_result.train_dyemat[keep_rows]

        assert train_radmat.shape == shape

        train_result = nn(
            test_nn_params.use_gmm,
            sim_result,
            radmat=train_radmat,
            true_dyemat=train_dyemat,
            progress=progress,
        )
        train_result.true_pep_iz = sim_result.train_true_pep_iz
        train_result.true_pep_iz = ArrayResult(
            filename="train_true_pep_iz",
            shape=(shape[0] * shape[1],),
            dtype=IndexType,
            mode="w+",
        )
        train_result.true_pep_iz[:] = np.repeat(
            np.arange(shape[0]).astype(IndexType), shape[1]
        )
        check.t(train_result.true_pep_iz, ArrayResult)
        check.t(train_result.pred_pep_iz, ArrayResult)

        call_bag = CallBag(
            true_pep_iz=train_result.true_pep_iz.arr(),
            pred_pep_iz=train_result.pred_pep_iz.arr(),
            scores=train_result.scores.arr(),
            prep_result=prep_result,
            sim_result=sim_result,
        )

        if pipeline is not None:
            pipeline.set_phase(4, n_phases)

        train_result.peps_pr = call_bag.pr_curve_by_pep(progress=progress)

        if pipeline is not None:
            pipeline.set_phase(5, n_phases)

        train_result.peps_pr_abund = call_bag.pr_curve_by_pep_with_abundance(
            progress=progress
        )

    else:
        train_result = {k: None for k in test_result.keys()}

    def rename(d, prefix):
        return {f"{prefix}{k}": v for k, v in d.items()}

    return TestNNResult(
        params=test_nn_params,
        **rename(test_result, "test_"),
        **rename(train_result, "train_"),
    )
