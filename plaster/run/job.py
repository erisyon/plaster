"""
Jobs are:
    folders under ./jobs_folder
    Created by generators (pgen)
    Jobs contain sub-folders called "runs"
    Runs contain plaster_run.yaml files and may contain plaster_output/ folders

This module organizes all-run results.
"""

import pandas as pd
from plaster.run.run import RunResult
from plaster.run.survey_v2.survey_v2_result import SurveyV2Result
from plaster.tools.assets import assets
from plaster.tools.log.log import debug
from plaster.tools.schema import check
from plaster.tools.zap import zap
from plumbum import local
from tqdm.auto import tqdm


class JobResult:
    """
    JIT loads results
    """

    def __init__(self, job_folder, include_manifest=True):
        self.job_folder = assets.validate_job_folder(
            job_folder, search_if_not_present=True
        )
        self._run_results = {
            run_folder.name: RunResult(run_folder, include_manifest=include_manifest)
            for run_folder in self.job_folder
            if run_folder.is_dir() and "run_manifest.yaml" in run_folder
        }

    @property
    def n_runs(self):
        return len(self._run_results)

    @property
    def runs(self):
        return list(self._run_results.values())

    def run_by_name(self, run_name):
        return self._run_results[run_name]

    def __getattr__(self, key):
        return self._run_results[key]

    def __getitem__(self, item):
        return list(self._run_results.values())[item]

    def all_dfs(self, fn, parallel=False):
        """
        Run fn on every run, assert that each returns af DataFrame
        and then pd.concat all the results into one adding a run_i
        column to that DataFrame.

        Example:
            df = job.all_dfs(lambda run: run.prep.pros())
        """
        df_list = []
        if parallel:

            def wrap_fn(run, run_i):
                res_df = fn(run)
                assert isinstance(res_df, pd.DataFrame)
                res_df["run_i"] = run_i
                res_df["run_name"] = run.manifest.run_name
                return res_df

            work_orders = [
                {"fn": wrap_fn, "args": [run, run_i]}
                for run_i, run in enumerate(self._run_results.values())
            ]

            # TODO: it would be nice to integrate this progress stuff into zap as an optional argument
            progress = tqdm(total=len(work_orders))

            def progress_callback(i, j, retry):
                if not retry:
                    progress.update()

            df_list = zap.work_orders(work_orders, _progress=progress_callback,)

            progress.close()
        else:
            for run_i, run in enumerate(self._run_results.values()):
                res_df = fn(run)
                assert isinstance(res_df, pd.DataFrame)
                res_df["run_i"] = run_i
                res_df["run_name"] = run.manifest.run_name
                df_list += [res_df]
        return pd.concat(df_list).reset_index(drop=True)

    def all_lists(self, fn):
        """
        Run fn on every run and return a list

        Example:
            bags = job.all_lists(lambda run: run.test_rf_call_bag())
        """
        return [fn(run) for run in self._run_results.values()]

    def all(self, fn):
        """
        Run fn on every run. No return.

        Example:
            job.all(lambda run: run.test_rf_call_bag())
        """
        [fn(run) for run in self._run_results.values()]

    """
    I'm adding these helpers below so that notebooks contain boilerplate code that
    is easy to read and modify by end-users.  Also, I originally thought I'd move
    these out to keep "Job" very clean/sparse as a generic container, but as I work
    on more reporting templates, it becomes clear that you really do want to look
    at jobs as a whole, and you want lots of easy ways to compare the runs they
    contain.  This is the job for something higher level than a run.  It is the
    job for ... a JobResult!
    """

    def set_pros_of_interest(self, protein_ids=[]):
        """
        protein_ids: a (possibly empty) list of protein_ids
        """
        check.t(protein_ids, list)
        self.all(lambda run: run.prep.set_pros_of_interest(protein_ids=protein_ids))

    def get_pros_of_interest(self):
        # Note: this copy() is here to avoid the SettingWithCopyWarning from Pandas
        # since otherwise we're getting back a slice and then setting run_i on it
        # within add_dfs..
        return self.all_dfs(lambda run: run.prep.pros__in_report().copy())

    def set_pro_ptm_locs(self, protein_id, ptms):
        """
        protein_id: id of protein
        ptms: ;-delimited list of ptm locations, or empty string for no ptms
        """
        self.all(
            lambda run: run.prep.set_pro_ptm_locs(protein_id=protein_id, ptms=ptms)
        )

    def get_pro_ptm_locs(self, protein_id):
        """
        Returns the ptm list for the given protein_id.
        Note that this information is stored in a DataFrame maintained on a per-run
        basis, so we sanity check here that ptms reported by all runs are the same.
        """
        ptms_by_run = self.all_lists(
            lambda run: run.prep.get_pro_ptm_locs(protein_id=protein_id)
        )
        check.affirm(
            all([ptms_by_run[0] == p for p in ptms_by_run[1:]]),
            "PTMs differ in runs!",
            ValueError,
        )
        return ptms_by_run[0]

    def peps_prs_report_df(
        self,
        include_poi_only=False,
        include_ptm_only=False,
        force_compute_prs=False,
        pr_with_abundance=False,
        classifier=None,
    ):
        """
        Create a df that contains prs values for every peptide across every run
        for proteins that are marked as "proteins of interest".
        Optionally only do this for peptides that contain ptm locations.

        This does not expose the include_decoys param, which defaults to False.
        """
        return self.all_dfs(
            lambda run: run.peps_prs_report_df(
                ptm_peps_only=include_ptm_only,
                in_report_only=include_poi_only,
                force_compute_prs=force_compute_prs,
                pr_with_abundance=pr_with_abundance,
                classifier=classifier,
            ),
            parallel=True,
        )

    @staticmethod
    def _best_prec_at_recall(df, filters):
        """
        Get best precision run(s) for min_recall given a groupby on peptide
        """

        df.reset_index()  # so that index is for this groupby() only

        # Get rows that meet the min_recall threshold
        df_min_recall = df[df.recall >= filters.min_recall]

        if len(df_min_recall) > 0:
            # note drop_duplicates because we're interested in the best n distinct runs
            df_min_recall = df_min_recall.sort_values(
                ["prec", "recall"], ascending=[False, False]
            ).drop_duplicates("run_i")
            return df_min_recall.iloc[: filters.n_best_runs]
        else:
            # This group has no recall >= min_recall, so return entry with max recall.
            return df.drop_duplicates("run_i").iloc[: filters.n_best_runs]

    @staticmethod
    def get_best_precision_runs_for_peptides(all_runs_pr_df, filters):
        """
        IMPORTANT: this function is for comparing runs that all use the
        same protease (or no protease) such that the peptides for the runs
        are identical -- we rely on the pep_iz being the same for each run.
        This is the case for jobs like MHC, but could also be the case if
        you've decided on a protease and want to compare label schemes.

        Given a Dataframe containing prs vales for every peptide across every run,
        select the top_n best runs for each peptide of interest.  "Best" means
        best precision available given the filtering conditions passed in filters.
        The "primary" filter is recall, such that we are saying, "I need recall of
        at least X. Other filters may be applied
        to constrain the selection.


        all_runs_pr_df: the df produced by peps_prs_report_df() above, which must
                        contain at least prec,recall,P2 for each pep_i
                        See peps_prs_report_df()

        filters: a Munch containing constraints on the selection.
             Ex:
                 filters = Munch(
                    min_recall=0.1,            # float
                    max_pep_len=None,          # None or int
                    allow_proline_at_2=True,   # True or False
                 )

        Returns:
                 (1) a df that is a subset of rows from all_runs_pr_df, which may be
                 passed directly to plotting routines to visualize full PR for each
                 peptide.
        """

        df = (
            all_runs_pr_df.copy()
        )  # copy() gets rid of SettingWithCopyWarning, but is it necessary?

        # Run include
        if filters.include_runs is not None and len(filters.include_runs) > 0:
            df = df[df.run_name.isin(filters.include_runs)]

        # Run exclude
        if filters.exclude_runs is not None and len(filters.exclude_runs) > 0:
            df = df[~df.run_name.isin(filters.exclude_runs)]

        # Proline at position 2 creates some challenges for our chemistry.
        #
        if not filters.allow_proline_at_2:
            df = df.loc[df.P2 == False]

        # Look at a subset of peptides if desired
        #
        if filters.peptide_subset is not None and filters.peptide_subset != []:
            column = "pep_i" if isinstance(filters.peptide_subset[0], int) else "seqstr"
            df = df[df[column].isin(filters.peptide_subset)]

        # Only include unique peptides
        #
        if filters.unique_peps is True:
            df = df[df.flu_count == 1]

        # Find the best run for each peptide.
        #
        if len(df) > 0:
            best_runs_pr = (
                df.groupby("pep_i")
                .apply(lambda df: JobResult._best_prec_at_recall(df, filters))
                .reset_index(drop=True)
            )
            best_runs_pr = best_runs_pr.sort_values(
                by=["prec", "recall"], ascending=[False, False]
            ).reset_index(drop=True)
        else:
            best_runs_pr = None

        return best_runs_pr

    @staticmethod
    def get_best_precision_runs_for_ptms(all_runs_pr_df, filters):
        """
        Given a Dataframe containing prs vales for [some,all] peptides across every run,
        select the top_n best runs for each ptm location of interest.  "Best" means
        best precision available given the filtering conditions passed in filters.
        The "primary" filter is recall, such that we are saying, "I need recall of
        at least X.  Now show me the best precision peptides across all runs that
        contain the PTM locations I'm interested in."  Other filters may be applied
        to constrain the selection.


        all_runs_pr_df: the df produced by peps_prs_report_df() above, which must
                        contain at least prec,recall,P2,ptm for pep_iz that will be
                        searched.  See peps_prs_report_df()

        filters: a Munch containing constraints on the selection.
             Ex:
                 filters = Munch(
                    min_recall=0.1,            # float
                    max_pep_len=None,          # None or int
                    max_ptms_per_pep=None,     # None or int
                    allow_proline_at_2=True,   # True or False
                 )

        Returns: a tuple containing:
                 (1) a df that is a subset of rows from all_runs_pr_df, which may be
                 passed directly to plotting routines to visualize full PR for each
                 peptide/PTM.  Note that some 'helper' columns are added by this
                 function: pep_len, n_pep_ptms
                 (2) a list (possibly empty) of PTM locations that survived filtering
                 (3) a list (possibly empty) of PTM locations that were removed by filtering
        """

        # We'll operate on all rows of the df grouped by ptm, so ensure
        # all rows have a non-empty ptm.
        df = all_runs_pr_df[
            all_runs_pr_df.ptm.astype(bool)
        ].copy()  # copy() gets rid of SettingWithCopyWarning, but is it necessary?

        # Run include
        if filters.include_runs is not None and len(filters.include_runs) > 0:
            df = df[df.run_name.isin(filters.include_runs)]

        # Run exclude
        if filters.exclude_runs is not None and len(filters.exclude_runs) > 0:
            df = df[~df.run_name.isin(filters.exclude_runs)]

        # Proline at position 2 creates some challenges for our chemistry.
        #
        if not filters.allow_proline_at_2:
            df = df.loc[df.P2 == False]

        # Filter by peptide length and max number of PTMs per peptide.
        # Leave these columns in the returned df as they are interesting/useful.
        #
        df["pep_len"] = df.pep_stop - df.pep_start
        if filters.max_pep_len is not None:
            df = df[df.pep_len <= filters.max_pep_len]

        # This should now be set in prep.peps__ptms
        # df["n_pep_ptms"] = df.apply(
        #     lambda x: len(x.pro_ptm_locs.split(";") if x.pro_ptm_locs else 0), axis=1
        # )

        if filters.max_ptms_per_pep is not None:
            df = df[df.n_pep_ptms <= filters.max_ptms_per_pep]

        # Look at a subset of ptm if desired
        #
        if filters.ptm_subset is not None and filters.ptm_subset != []:
            str_ptms = [str(ptm) for ptm in filters.ptm_subset]
            df = df[df.ptm.isin(str_ptms)]

        # It's possible that filtering completely removes our access to some PTM locations.
        # Create lists of those remaining and those removed to be returned to the caller.
        #
        remain_ptms = set(df.ptm.unique())
        removed_ptms = set(all_runs_pr_df.ptm.unique()) - remain_ptms

        # Find the best run for each PTM location of interest.
        #
        if len(df) > 0:
            df.ptm = df.ptm.astype(int)
            best_runs_pr = (
                df.groupby("ptm")
                .apply(lambda df: JobResult._best_prec_at_recall(df, filters))
                .reset_index(drop=True)
                .sort_values("ptm")
            )
        else:
            best_runs_pr = None

        return (
            best_runs_pr,
            sorted(map(int, remain_ptms)),
            sorted(map(int, removed_ptms)),
        )

    @staticmethod
    def get_best_precision_runs_for_pros(all_runs_pr_df, filters):
        """
        Given a Dataframe containing prs vales for [some, all] peptides across every run,
        select the top_n best runs for identifying proteins.  "Best" means
        best precision available given the filtering conditions passed in filters.
        The "primary" filter is recall, such that we are saying, "I need recall of
        at least X.  See other "get_precsion_precision_runs_for_xxxx" fns above.

        For identifying proteins, we want at least 1 very high precision peptide,
        so the approach we'll take is to groupby pro_i and find the n_best_runs
        which produce the highest precision peptides.

        all_runs_pr_df: the df produced by peps_prs_report_df() above, which must
                        contain at least prec,recall,P2,ptm for pep_iz that will be
                        searched.  See peps_prs_report_df()

        filters: a Munch containing constraints on the selection.
             Ex:
                 filters = Munch(
                    min_recall=0.1,            # float
                    max_pep_len=None,          # None or int
                    max_dyes_per_channel=None, # None or int
                    allow_proline_at_2=True,   # True or False
                 )

        Returns: a tuple containing:
                 (1) a df that is a subset of rows from all_runs_pr_df, which may be
                 passed directly to plotting routines to visualize full PR for each
                 protein based on best peptide(s).  Note that some 'helper' columns
                  are added by this function: pep_len
        """

        df = (
            all_runs_pr_df.copy()
        )  # copy() gets rid of SettingWithCopyWarning, but is it necessary?

        # Run include
        if filters.include_runs is not None and len(filters.include_runs) > 0:
            df = df[df.run_name.isin(filters.include_runs)]

        # Run exclude
        if filters.exclude_runs is not None and len(filters.exclude_runs) > 0:
            df = df[~df.run_name.isin(filters.exclude_runs)]

        # Protein subset
        if filters.pro_subset is not None and len(filters.pro_subset) > 0:
            df = df[df.pro_id.isin(filters.pro_subset)]

        # Proline at position 2 creates some challenges for our chemistry.
        #
        if not filters.allow_proline_at_2:
            df = df.loc[df.P2 == False]

        # Max dyes per channel
        #
        if filters.max_dyes_per_ch is not None:
            df = df[df.n_dyes_max_any_ch <= filters.max_dyes_per_ch]

        # Filter by peptide length
        # Leave pep_len in the returned df
        #
        df["pep_len"] = df.pep_stop - df.pep_start
        if filters.max_pep_len is not None:
            df = df[df.pep_len <= filters.max_pep_len]

        # Find the best run for each protein.
        #
        if len(df) > 0:
            best_runs_pr = (
                df.groupby("pro_i")
                .apply(lambda df: JobResult._best_prec_at_recall(df, filters))
                .reset_index(drop=True)
            )
        else:
            best_runs_pr = None

        return best_runs_pr

    def get_nn_stats_df(self, filters=None):
        """
        This is 'early' in development, but produces useful results.

        The idea is that the job collects survey statistics from each run and depending
        on what the objective is (protein identification, protein/peptide coverage, ptm)
        we return the relevant stats per run.

        Since this logic is really intimately tied to the idea of a "survey", I think
        this logic should end up in the NNSurveyResult, perhaps as a static,
        or at least in that module as a free function.

        It's here along with much other job-related reporting because it's easy
        to remember when you're editing in a notebook that top-level reporting stuff
        lives in the job object:  job.do_some_report(filters) -- and if you can't
        remember, just look in the JobResult to see what it can do.
        """

        filters = filters or SurveyV2Result.defaults

        def _filter_runs(df):
            # A last minute addition to survey is the ability to include
            # or exclude particular runs by name, and this is done *after*
            # all runs have contributed to the df because it is simplest and
            # cleanest to do this -- though it means work is done by each
            # run even it if will not be used, and it means filters that
            # rightly belong to NNSurveyResult are being acted on here
            # at a higher level.  This is a convenience to allow the user
            # to look more closely at a single run.

            # Run include
            if filters.include_runs is not None and len(filters.include_runs) > 0:
                df = df[df.run_name.isin(filters.include_runs)]
            # Run exclude
            if filters.exclude_runs is not None and len(filters.exclude_runs) > 0:
                df = df[~df.run_name.isin(filters.exclude_runs)]
            return df

        if filters.objective == "protein_id" or filters.objective == "ptms":
            # To identify proteins, we're interested in the best single peptide
            # that can be used to identify the protein.  So find the one(s) with
            # the max distance from all other peptides in the domain.  For ptms,
            # the process is very similar, but we are likely interested in multiple
            # peptides from the same protein.  The details are managed in the call
            # to max_nn_dist_peps() which looks at the objective in deciding which
            # peps to include in the returned df.

            df = self.all_dfs(
                lambda r: r.survey_v2.max_nn_dist_peps(prep=r.prep, filters=filters)
            )

            df = _filter_runs(df)

            if filters.multi_peptide_metric is None:
                # We are ranking based on single peptides only, so sort on nn_dist to
                # see the peptides with max nn_dist -- most isolated, easiest to classify.
                df = df.sort_values(
                    by=["nn_dist", "nn_coverage"], ascending=[False, False]
                )
                final_sort_keys = ["sort_runs", "nn_dist", "nn_coverage"]
                final_sort_ascend = [True, False, False]
            else:
                # Here multi_peptide_metric tells us which to sort first by, but we
                # want to sort by the other(s) as well.  Since we're trying to id
                # multiple peptides, we somehow must take into account the nn_dist
                # for multiple "best" peptides -- so average, or take the best, etc.
                sort_keys = ["nn_dist_avg", "nn_dist_min", "nn_coverage"]
                sort_keys.insert(
                    0,
                    sort_keys.pop(
                        sort_keys.index("nn_" + filters.multi_peptide_metric)
                    ),
                )
                df = df.sort_values(by=sort_keys, ascending=[False] * len(sort_keys))
                final_sort_keys = ["sort_runs", "nn_rank", "nn_dist", "nn_coverage"]
                final_sort_ascend = [True, True, False, False]

            # Keep the top n runs (n_best_schemes), and use those run numbers, in order,
            # as the primary sort key so runs are grouped together in the output.
            run_iz_to_keep = list(df.run_i.unique()[: filters.n_best_schemes])
            df = df[df.run_i.isin(run_iz_to_keep)]

            df["sort_runs"] = df.apply(lambda r: run_iz_to_keep.index(r.run_i), axis=1)
            df = df.sort_values(
                by=final_sort_keys, ascending=final_sort_ascend,
            ).reset_index(drop=True)
            return df.drop("sort_runs", axis=1)

        if filters.objective == "coverage":
            # For maximum coverage of some domain, it does not depend on a single
            # peptide performance but rather how much of the domain is covered by
            # peptides with unique fluorosequences.  In this case we'll return
            # stats for each run (one row per run) sorted by coverage.
            stats = self.all_lists(
                lambda r: r.survey_v2.nn_stats(r.prep, filters=filters)
            )
            run_info = [(i, r.run_name) for i, r in enumerate(self.runs)]
            df = pd.DataFrame(
                [r + s for r, s in zip(run_info, stats)],
                columns=["run_i", "run_name", "nn_uniques", "nn_coverage", "nn_dist"],
            ).sort_values(
                by=["nn_coverage", "nn_dist", "nn_uniques"],
                ascending=[False, False, False],
            )
            df = _filter_runs(df)
            return df[: filters.n_best_schemes]

        raise ValueError(f"Bad objective value: {filters.objective}")


class MultiJobResult(JobResult):
    """
    Subclass handles loading runs from multiple jobs to be viewed as one job.
    This relies on the currently true fact that once the runs have been loaded,
    all operations performed by JobResult are simply operations on a list of
    runs, agnostic of where they came from.

    Note that if identically-named runs exist across job folders, those encountered
    last (via job folders coming later in the list) will get put into _run_results.
    """

    def __init__(self, job_folders, include_manifest=True):
        check.t(job_folders, list)

        self.job_folder = "MultiJobResult has multiple folders in job_folders"
        self.job_folders = []
        self._run_results = {}
        for job_folder in job_folders:
            job_folder = assets.validate_job_folder(job_folder)
            self.job_folders += [job_folder]
            self._run_results.update(
                {
                    run_folder.name: RunResult(
                        run_folder, include_manifest=include_manifest
                    )
                    for run_folder in job_folder
                    if run_folder.is_dir() and "run_manifest.yaml" in run_folder
                }
            )
