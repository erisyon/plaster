"""
Helpers for making task blocks

Guidelines:
    * Each generator is a pure function.
    * There is minimal parameterization on each helper; the correct behavior
      of a caller is to MODIFY the returned value to patch up custom odds and ends
      to avoid a glut of parameter passing.
"""

from munch import Munch
from plaster.run.error_model import ErrorModel, GainModel, ChGainModel
from plaster.run.sigproc_v2 import sigproc_v2_common
from plaster.run.sim_v1.sim_v1_params import SimV1Params
from plaster.run.sim_v2.sim_v2_params import SimV2Params
from plaster.tools.aaseq.aaseq import aa_list_to_str
from plaster.tools.log.log import debug
from plaster.tools.schema import check
from plaster.tools.utils import utils


def erisyon(generator_name="", sample="", run_name="", **kwargs):
    """
    This method is an example of name-space reservation.
    Certain tools things like proteases, aa_list are emitted here
    not because all generators have those but to prevent multiple
    generators for having *different* meanings for these fields.
    """
    check.t(generator_name, str)
    check.t(sample, str)

    return Munch(
        _erisyon=Munch(
            run_name=run_name,
            run_pk=utils.random_str(8),
            sample=sample,
            generator_name=generator_name,
            **kwargs,
        )
    )


def ims_import(src_dir, is_movie=False, n_cycles_limit=None):
    return Munch(
        ims_import=Munch(
            version="1.0",
            inputs=Munch(src_dir=src_dir),
            parameters=Munch(is_movie=is_movie, n_cycles_limit=n_cycles_limit),
        )
    )


def sigproc_v1():
    return Munch(
        sigproc_v1=Munch(
            version="1.0", inputs=Munch(ims_import="../ims_import"), parameters=Munch(),
        )
    )


def sigproc_v2_analyze(calibration_file):
    assert calibration_file is not None
    return Munch(
        sigproc_v2=Munch(
            version="1.0",
            inputs=Munch(ims_import="../ims_import"),
            parameters=Munch(
                calibration_file=calibration_file,
                mode=sigproc_v2_common.SIGPROC_V2_INSTRUMENT_ANALYZE,
            ),
        )
    )


def sigproc_v2_calib(calibration_file, mode):
    """
    TODO: subject_id
    """
    assert calibration_file is not None
    return Munch(
        sigproc_v2=Munch(
            version="1.0",
            inputs=Munch(ims_import="../ims_import"),
            parameters=Munch(calibration_file=calibration_file, mode=mode,),
        )
    )


def lnfit(sigproc_version):
    return Munch(
        lnfit=Munch(
            version="1.0",
            inputs=Munch(sigproc_dir=f"../sigproc_{sigproc_version}"),
            parameters=Munch(),
        )
    )


def prep(
    seqs,
    protease,
    decoy_mode,
    n_peps_limit=None,
    proteins_of_interest=None,
    n_ptms_limit=None,
    **kws,
):
    # TODO: rename all the stuff like pro_report or in_report to simply POI.
    # Originally this only affected reporting, but this is changing.
    if proteins_of_interest:
        # gen flag --protein_of_interest overrides all else
        pro_reports = [int(seq["id"] in proteins_of_interest) for seq in seqs]
    elif "in_report" in seqs[0]:
        # but POI may also be specified in a --protein_csv
        pro_reports = [s["in_report"] for s in seqs]
    else:
        # else no proteins marked "of interest"
        pro_reports = [0] * len(seqs)

    return Munch(
        prep=Munch(
            version="1.0",
            inputs=Munch(),
            parameters=Munch(
                n_peps_limit=n_peps_limit,
                n_ptms_limit=n_ptms_limit,
                protease=protease if protease is None else protease.split("+"),
                decoy_mode=decoy_mode,
                proteins_of_interest=proteins_of_interest,
                **kws,
                proteins=[
                    Munch(
                        name=seq["id"],
                        report=report,
                        sequence=aa_list_to_str(seq["seqstr"], spaces=10),
                        ptm_locs=seq["ptm_locs"] if "ptm_locs" in seq else "",
                        abundance=seq.get("abundance"),
                    )
                    for seq, report in zip(seqs, pro_reports)
                ],
            ),
        )
    )


def sim_v1(aa_list, err_set, **sim_kws):
    if isinstance(err_set, ErrorModel):
        error_model = err_set
    else:
        error_model = ErrorModel.from_err_set(err_set)
    assert sim_kws.get("n_edmans", 0) > 1
    n_pres = sim_kws.get("n_pres", 0)
    n_mocks = sim_kws.get("n_mocks", 0)
    assert (
        n_pres + n_mocks >= 1
    ), "You must include at least 1 pre or mock cycle to capture the initial image"
    return Munch(
        sim_v1=Munch(
            version="1.0",
            inputs=Munch(prep="../prep"),
            parameters=Munch(
                **SimV1Params.construct_from_aa_list(
                    aa_list, error_model=error_model, include_dfs=False, **sim_kws
                )
            ),
        )
    )


def sim_v2(aa_list, err_set, **sim_kws):
    if isinstance(err_set, ErrorModel):
        error_model = err_set
    else:
        error_model = ErrorModel.from_err_set(err_set)
    assert sim_kws.get("n_edmans", 0) > 1
    n_pres = sim_kws.get("n_pres", 0)
    n_mocks = sim_kws.get("n_mocks", 0)
    assert (
        n_pres + n_mocks >= 1
    ), "You must include at least 1 pre or mock cycle to capture the initial image"
    return Munch(
        sim_v2=Munch(
            version="1.0",
            inputs=Munch(prep="../prep"),
            parameters=Munch(
                **SimV2Params.construct_from_aa_list(
                    aa_list, error_model=error_model, include_dfs=False, **sim_kws
                )
            ),
        )
    )


def survey_v2():
    return Munch(
        survey_v2=Munch(
            version="1.0",
            inputs=Munch(prep="../prep", sim_v2="../sim_v2"),
            parameters=Munch(),
        )
    )


def train_rf():
    return Munch(
        train_rf=Munch(
            version="1.0",
            inputs=Munch(sim_v1="../sim_v1"),
            parameters=Munch(
                n_estimators=10,
                min_samples_leaf=50,
                max_depth=None,
                max_features="auto",
                max_leaf_nodes=None,
            ),
        )
    )


def test_rf():
    return Munch(
        test_rf=Munch(
            version="1.0",
            inputs=Munch(sim_v1="../sim_v1", train_rf="../train_rf", prep="../prep"),
            parameters=Munch(include_training_set=False, keep_all_class_scores=False),
        )
    )


def ptm_train_rf(ptm_labels, proteins_of_interest):
    return Munch(
        ptm_train_rf=Munch(
            version="1.0",
            inputs=Munch(prep="../prep", sim="../sim_v1", train_rf="../train_rf"),
            parameters=Munch(
                proteins_of_interest=proteins_of_interest, ptm_labels=ptm_labels
            ),
        )
    )


def ptm_test_rf():
    return Munch(
        ptm_test_rf=Munch(
            version="1.0",
            inputs=Munch(
                prep="../prep",
                sim="../sim_v1",
                train_rf="../train_rf",
                ptm_train_rf="../ptm_train_rf",
            ),
            parameters=Munch(),
        )
    )


def classify_rf(train_relative_path, sigproc_relative_path, sim_relative_path):
    return Munch(
        classify_rf=Munch(
            version="1.0",
            inputs=Munch(
                sim_v1=sim_relative_path,
                train_rf=train_relative_path,
                sigproc_v1=sigproc_relative_path,
            ),
            parameters=Munch(),
        )
    )


def nn_v1(**kws):
    return Munch(
        nn_v1=Munch(
            version="1.0",
            inputs=Munch(sim_v1="../sim_v1", prep="../prep"),
            parameters=Munch(**kws),
        )
    )


def calib_nn_v1(
    mode, n_pres, n_mocks, n_edmans, dye_names, scope_name, channels,
):
    return Munch(
        calib=Munch(
            version="1.0",
            inputs=Munch(sigproc="../sigproc_v1"),
            parameters=Munch(
                mode=mode,
                n_pres=n_pres,
                n_mocks=n_mocks,
                n_edmans=n_edmans,
                dye_names=dye_names,
                scope_name=scope_name,
                channels=channels,
            ),
        )
    )


def nn_v2(sigproc_relative_path, err_set, **kws):
    n_channels = len(err_set.dye_beta)
    task = Munch(
        nn_v2=Munch(
            version="1.0",
            inputs=Munch(prep="../prep", sim_v2="../sim_v2"),
            parameters=Munch(
                gain_model=GainModel(
                    row_k_beta=1.0,  # TODO: this needs to be added as a parameter
                    row_k_sigma=0.16,  # TODO: this needs to be added as a parameter
                    channels=[
                        ChGainModel(
                            beta=err_set.dye_beta[ch_i],
                            sigma=err_set.dye_sigma[ch_i],
                            zero_beta=err_set.dye_zero_beta[ch_i],
                            zero_sigma=err_set.dye_zero_sigma[ch_i],
                        )
                        for ch_i in range(n_channels)
                    ],
                ).asdict(),
                **kws,
            ),
        )
    )

    if sigproc_relative_path is not None:
        task.nn_v2.inputs.sigproc = sigproc_relative_path
        task.nn_v2.parameters.include_sigproc = True

    return task
