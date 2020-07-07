from plaster.run.base_result import BaseResult
from plaster.run.lnfit.lnfit_params import LNFitParams
import pandas as pd
import pickle
import re


class LNFitResult(BaseResult):
    """
    This Result class is a little different than others in that it tries to
    manage multiple lnfits that may occur in a single Run.  The Run class now
    loads things by task_name instead of task, so we could change this to
    only deal with a single task.
    """

    name = "lnfit"
    filename = "lnfit.pkl"

    required_props = dict(
        params=LNFitParams, photometry_rows=int, dye_on_threshold=int, did_fit=bool,
    )

    def _task_folders(self, lnfit_taskname=None):
        """
        Multiple lnfit tasks may be run in a single Run.  These tasks can be
        found by examining the manifest.
        """
        task_folders = []
        for task_name, task_block in self.run.manifest.tasks.items():
            if task_name == "lnfit" or (
                "task" in task_block and task_block.task == "lnfit"
            ):
                # This is an lnfit task.  Was a specific one requested?
                if lnfit_taskname is not None and lnfit_taskname != task_name:
                    continue

                task_folders += [task_name]

        assert task_folders
        assert all([(self.run.run_output_folder / f).is_dir() for f in task_folders])

        return task_folders

    def html_files(self):
        task_folders = self._task_folders()
        lnfit_html_files = []
        for f in task_folders:
            lnfit_html_files += [(self.run.run_output_folder / f // "*.html")]
        return lnfit_html_files

    def lnfit_intermediates_df(self, lnfit_taskname):
        """
        Returns the "full" intermediates output of lnfit as a DataFrame.
        If lnfit_taskname is not specified, this includes potentially
        multiple lnfit tasks from this run.
        """
        task_folders = self._task_folders(lnfit_taskname=lnfit_taskname)
        lnfit_pkls = []
        for f in task_folders:
            pkls = (
                self.run.run_output_folder
                / f
                // "track_photometries.csv_??????_ch?_INTERMEDIATES_v2.pkl"
            )
            # If you run lnfit more than once without clearing the folder, it will
            # produce a new set of output files with a new timestamp hashtag.
            # Luckily these hashtags sort alphabetically - take the latest one.
            if pkls:
                lnfit_pkls += [sorted(pkls)[0]]

        # Create one big DataFrame from all pkls
        #
        dfs = []
        for pickle_filename in lnfit_pkls:
            p = pickle.load(open(pickle_filename, "rb"), encoding="latin1")
            a = all_fit_info = p[1][3]  # "all_fit_info", or "plf_results"
            d = dict(
                channel=[e[0] for e in all_fit_info],
                field=[e[1] for e in all_fit_info],
                peak_i=[
                    e[2] for e in all_fit_info
                ],  # IMPORTANT HACK: see _alex_track_photometries_csv
                # y=[e[3] for e in all_fit_info],
                row=[e[4] for e in all_fit_info],
                category=[e[5] for e in all_fit_info],
                intensities=[e[6] for e in all_fit_info],
                signal=[e[7] for e in all_fit_info],
                is_zero=[e[8] for e in all_fit_info],
                best_seq=[e[9] for e in all_fit_info],
                lmii=[e[10] for e in all_fit_info],
                best_score=[e[11] for e in all_fit_info],
                best_intensity_score=[e[12] for e in all_fit_info],
                starting_intensity=[e[13] for e in all_fit_info],
            )
            dfs += [pd.DataFrame(d)]
        df = pd.concat(dfs, ignore_index=True)
        return df

    def lnfit_bestseq_df(self, lnfit_taskname=None):
        """
        Returns a minimal DataFrame from lnfit intermediate results by removing
        rows that failed to find a best fit sequence, converting channel to
        0-based int, and concat-ing the sequence for readability.
        """
        df = self.lnfit_intermediates_df(lnfit_taskname)
        df = df[~df.best_seq.isnull()][["peak_i", "best_seq", "channel"]]
        df.best_seq = df.best_seq.apply(lambda x: "".join(map(str, x)))
        df["channel_i"] = (df.channel.str[2]).astype(int) - 1
        df = df.drop(["channel"], axis=1)
        df["best_seq_count"] = df.groupby("best_seq")["best_seq"].transform("count")
        df = df.sort_values(by=["best_seq_count"], ascending=False).reset_index()
        return df

    def __repr__(self):
        try:
            return (
                f"LNFitResult n_rows:{len(self.photometry_rows)} did_fit:{self.did_fit}"
            )
        except:
            return "LNFitResult"
