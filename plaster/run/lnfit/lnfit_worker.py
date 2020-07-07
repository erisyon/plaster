import numpy as np
import pandas as pd
import os
from plaster.tools.pipeline.pipeline import PipelineTask
from plaster.tools.utils import utils
from plumbum import local, FG
from plumbum.commands.processes import CommandNotFound
from plaster.run.lnfit.lnfit_result import LNFitResult
from plaster.tools.log.log import debug


class LnFitError(Exception):
    ignore_traceback = True


def _alex_track_photometries_csv(sigproc_result, threshold):
    """
    Convert plaster radiometry to alex/pflib photometry csv format so that plaster
    sigprocv2 results can be fit by the pflib lognormal fitter.

    # Example from Alex's file
    CHANNEL,FIELD,H,W,CATEGORY,FRAME 0,FRAME 1,FRAME 2,FRAME 3,FRAME 4,FRAME 5,FRAME 6,FRAME 7,FRAME 8,FRAME 9
    ch1,0,34.55,492.45,"(False, True, False, True, True, True, False, False, False, False)",8462.0,-469.5,9661.5,14135.0,-9409.5,-1875.0,-4356.0,-1565.5,-5543.5,-1936.5

    Returns: CSV as str
    """

    df = pd.DataFrame()

    for ch in range(sigproc_result.n_channels):
        rad = sigproc_result.signal_radmat()[:, ch, :]
        data_df = pd.DataFrame(
            rad, columns=[f"FRAME {i}" for i in range(sigproc_result.n_cycles)]
        )
        rows = rad.shape[0]
        peaks = sigproc_result.peaks()
        pref_df = pd.DataFrame(
            dict(
                CHANNEL=[f"ch{ch + 1}"] * rows,
                FIELD=[peaks.loc[i].field_i for i in range(rows)],
                H=[i for i in range(rows)],  # IMPORTANT HACK: this is the peak number!
                W=[i for i in range(rows)],
                # H=np.random.uniform(0, 512, size=(rows,)).astype(int),
                # W=np.random.uniform(0, 512, size=(rows,)).astype(int),
            )
        )
        above_thresh = np.where(rad > threshold, True, False)
        category = np.apply_along_axis(
            lambda row: np.array(
                "(" + ", ".join(row.astype(str).tolist()) + ")", object
            ),
            1,
            above_thresh,
        )
        category_df = pd.DataFrame(category.astype(str), columns=["CATEGORY"])
        _df = pd.concat([pref_df, category_df, data_df], axis=1)
        df = pd.concat([df, _df], axis=0)

    return df.to_csv(index=False, float_format="%.0f")


def lnfit(lnfit_params, sigproc_result):

    csv = _alex_track_photometries_csv(sigproc_result, lnfit_params.dye_on_threshold)

    # This photometry_filename will get mounted into the container
    photometry_filename = "track_photometries.csv"
    utils.save(photometry_filename, csv)

    if not lnfit_params.photometry_only:
        # If we're running in a docker context then the path will start with
        # /app, which we need to substitute for the real host OS path.
        # This has to come from the environment since it might vary from host to host
        data_folder = os.environ.get("HOST_PLASTER_DATA_FOLDER", "./jobs_folder")
        data_folder = os.path.join(data_folder, "")  # Adds a slash if needed
        lnfit_path = str(local.path(".")).replace("/app/jobs_folder/", data_folder)

        def run_docker_command(command):
            local["bash"]["-c", utils.get_ecr_login_string()] & FG

            aws_creds = []
            if local.env.get("ON_AWS", "0") == "0":
                aws_creds = [
                    f"--env",
                    f"AWS_ACCESS_KEY_ID={local.env['AWS_ACCESS_KEY_ID']}",
                    f"--env",
                    f"AWS_SECRET_ACCESS_KEY={local.env['AWS_SECRET_ACCESS_KEY']}",
                    f"--env",
                    f"AWS_DEFAULT_REGION={local.env['AWS_DEFAULT_REGION']}",
                ]

            local["docker"][
                [
                    f"run",
                    f"-it",
                    *aws_creds,
                    f"--mount",
                    f"type=bind,source={lnfit_path},target=/lnfit",
                    f"188029688209.dkr.ecr.us-east-1.amazonaws.com/alex:latest",
                    f"bash",
                    f"-c",
                    command,
                ]
            ] & FG(retcode=None)

        container_command = (
            f"cd /home/proteanseq "
            f"&& python ./pflib/lognormal_fitter_v2.py "
            f" {lnfit_params.lognormal_fitter_v2_params} "
            f" /lnfit/{photometry_filename} "
            f" >/lnfit/LN.OUT 2>/lnfit/LN.ERR"
        )

        try:
            run_docker_command(container_command)
        except CommandNotFound as e:
            raise LnFitError

    return LNFitResult(
        params=lnfit_params,
        photometry_rows=csv.count("\n") - 1,
        dye_on_threshold=lnfit_params.dye_on_threshold,
        did_fit=not lnfit_params.photometry_only,
    )
