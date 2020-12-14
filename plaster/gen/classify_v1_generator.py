from munch import Munch
from plaster.gen import task_templates
from plaster.gen.base_generator import BaseGenerator
from plaster.tools.log.log import debug, important
from plaster.tools.schema.schema import Schema as s
from plaster.tools.utils import utils
from plumbum import cli
from plaster.tools.log.log import current_file_and_line_str


class ClassifyV1Generator(BaseGenerator):
    """
    General-purpose generator for classifying peptides/proteins.
    May be used to search for one or more "needle" peptides.

    Assumptions:

    Generator-specific arguments:
    @--protein_of_interest="P10636-8"           # Only affects reporting downstream

    """

    # These schema are in general subsets of the "params" for different plaster tasks,
    # and for convenience in sharing among generators they are defined in BaseGenerator.
    # Its a bit arbitrary where some parameters end up, because they might be shared
    # by two different tasks that both get run as part of a classify run.  For example,
    # this classify generator supports runs that classify either just simulations, or
    # additionally actual data from a scope.  Both sims and scope runs need n_edmans,
    # n_mocks, n_pres.  But the schema for each cannot both contain these else we'll
    # pass duplicate key names into the schema below.

    schema = s(
        s.is_kws_r(
            **BaseGenerator.job_setup_schema.schema(),
            **BaseGenerator.protein_schema.schema(),
            **BaseGenerator.label_set_schema.schema(),
            **BaseGenerator.lnfit_schema.schema(),
            **BaseGenerator.scope_run_schema.schema(),
            **BaseGenerator.peptide_setup_schema.schema(),
            **BaseGenerator.sigproc_source_schema.schema(),
            **BaseGenerator.sigproc_v1_schema.schema(),
            **BaseGenerator.error_model_schema.schema(),
            **BaseGenerator.sim_schema.schema(),
            **BaseGenerator.classify_schema.schema(),
            **BaseGenerator.scheme_schema.schema(),
        )
    )

    defaults = Munch(
        n_edmans=10,
        n_pres=0,
        n_mocks=1,
        n_samples_train=5_000,
        n_samples_test=1_000,
        decoys="none",
        random_seed=None,
        nnF_v1=False,
        nn_v2=True,
        rf=False,
        sigproc_source=None,
        protein_of_interest=None,
        lnfit_name=None,
        lnfit_params=None,
        lnfit_dye_on_threshold=None,
        movie=False,
        radial_filter=None,
        peak_find_n_cycles=4,
        peak_find_start=0,
        anomaly_iqr_cutoff=95,
        # dye_beta=[7500.0],
        # dye_sigma=[0.16],
        n_ptms_limit=5,
        report_prec=[0.95, 0.9, 0.8],
    )

    def apply_defaults(self):
        super().apply_defaults()

        # Plumbum creates empty lists on list switches. This means
        # that the apply defaults doesn't quite work right.
        # TASK: Find a cleaner solution. For now hard-code
        # if len(self.err_dye_beta) == 0:
        #     self.err_dye_beta = self.defaults.dye_beta
        # if len(self.dye_sigma) == 0:
        #     self.dye_sigma = self.defaults.dye_sigma
        if len(self.report_prec) == 0:
            self.report_prec = self.defaults.report_prec

    def generate(self):

        self.report_section_user_config()

        sigproc_tasks = self.sigprocs_v1() or [{}]  # guarantee traverse loop once

        # TODO: 'default' reporting needs to be rethought.  Maybe we just employ
        # gen switch that says which report type.  The pattern that has developed
        # is that each project of any substance wants a special type of report.  These
        # projects are different enough that you always want to include custom stuff.
        # Presumably as we do more collabs/projects, they tend to group into a
        # handful of basic types.
        #
        # Bear in mind that we're in the classify generator, so all of these
        # refer to jobs that involve classification. (jobs like photobleaching
        # or other sigprocv2-only tasks don't -- those have their own hacky
        # report logic similar to what you'll see below).
        #
        # Currently those types are: 'standard' sigprocv2 with classify,
        # spike-in sigprocv2 with classify.
        #
        # VFS-only types: 'standard classify', PTM classify,
        # MHC classify (perhaps this is really standard classify, but is big, and
        # does not use a protease, and has all small uniform-length peptides)
        #
        # See all the hacky logic after these loops that patch together
        # a report by trying to deduce which of the above we're looking
        # at.
        #
        # Maybe we just need different generators instead of including
        # complex reporting logic?
        #
        # Etc.
        #

        # PTM, MHC, and PRO are the three classes of highest-level specialized reports
        # that report on all of the runs in a job taken together.  Whereas the default
        # report that comes out of classify will emit a long report with one section per
        # run, this became totally unwieldy when a job has 50+ (or hundreds!) of runs.
        # In that case you really only want a high-level report with a way to explore
        # the runs, and that's exactly what the specialized PTM, MHC, and PRO templates
        # are created for.  Here we try to cleverly deduce what kind of report we should
        # do based on whether there are PTMs present, Proteins-of-interest present, or
        # in the hackiest case, whether the sample or job name contains a given string.
        #
        # A PTM report is done if PTMs have been specified for any of the proteins
        ptm_report = any([pro.get("ptm_locs") for pro in self.protein])

        # A MHC-style report (which is special in that we know ahead of time that
        # the peptides are identical for all runs -- because we started with a list
        # of peptides -- so we can do lots of interesting comparisons that you can't
        # do when the peptides differ from run-to-run) is created for jobs which have
        # the string 'mhc' in their job-name or sample-name.  This needs to change,
        # but our Broad MHC project is the only one of this class for a year now.
        # This report is useful for any job that contains runs whose peptides are
        # identical -- this means either peptides were provided in the first place
        # and no protease was given to the "prep" task, or that only one protease,
        # and potentially lots of label schemes, is used.
        mhc_report = not ptm_report and (
            "mhc" in self.job.lower() or "mhc" in self.sample.lower()
        )

        # A protein-identification report is done if there are proteins of interest
        pro_report = (
            not ptm_report
            and not mhc_report
            and (
                bool(self.protein_of_interest)
                or any([pro.get("in_report") for pro in self.protein])
            )
        )

        run_descs = []
        for protease, aa_list, err_set in self.run_parameter_permutator():
            for sigproc_i, sigproc_v1_task in enumerate(sigproc_tasks):
                prep_task = task_templates.prep(
                    self.protein,
                    protease,
                    self.decoys,
                    proteins_of_interest=self.protein_of_interest,
                    n_ptms_limit=self.n_ptms_limit,
                )

                sim_v1_task = {}
                sim_v2_task = {}
                train_rf_task = {}
                test_rf_task = {}
                nn_v1_task = {}
                nn_v2_task = {}
                classify_rf_task = {}

                if self.rf:
                    train_rf_task = task_templates.train_rf()
                    test_rf_task = task_templates.test_rf()
                    if sigproc_v1_task:
                        classify_rf_task = task_templates.classify_rf(
                            sim_relative_path="../sim_v1",
                            train_relative_path="../train_rf",
                            sigproc_relative_path=f"../sigproc_v1",
                        )

                if self.nn_v1:
                    # note: same seed is used to generate decoys
                    nn_v1_task = task_templates.nn_v1()

                if self.nn_v2:
                    sim_v2_task = task_templates.sim_v2(
                        list(aa_list),
                        err_set,
                        n_pres=self.n_pres,
                        n_mocks=self.n_mocks,
                        n_edmans=self.n_edmans,
                        n_samples_train=self.n_samples_train,
                        n_samples_test=self.n_samples_test,
                    )
                    sim_v2_task.sim_v2.parameters.random_seed = self.random_seed
                    sigproc_relative_path = None
                    if sigproc_v1_task:
                        sigproc_relative_path = f"../sigproc_v1"

                    nn_v2_task = task_templates.nn_v2(
                        sigproc_relative_path=sigproc_relative_path,
                        err_set=err_set,
                        prep_folder="../prep",
                        sim_v2_folder="../sim_v2",
                    )

                if self.nn_v1 or self.rf:
                    sim_v1_task = task_templates.sim_v1(
                        list(aa_list),
                        err_set,
                        n_pres=self.n_pres,
                        n_mocks=self.n_mocks,
                        n_edmans=self.n_edmans,
                        n_samples_train=self.n_samples_train,
                        n_samples_test=self.n_samples_test,
                    )
                    sim_v1_task.sim_v1.parameters.random_seed = self.random_seed

                lnfit_task = self.lnfits("v2")

                e_block = self.erisyon_block(aa_list, protease, err_set)

                sigproc_suffix = (
                    f"_sigproc_{sigproc_i}" if len(sigproc_tasks) > 1 else ""
                )

                run_name = f"{e_block._erisyon.run_name}{sigproc_suffix}"
                if self.force_run_name is not None:
                    run_name = self.force_run_name

                run_desc = Munch(
                    run_name=run_name,
                    **e_block,
                    **prep_task,
                    **sim_v1_task,
                    **sim_v2_task,
                    **train_rf_task,
                    **test_rf_task,
                    **nn_v1_task,
                    **nn_v2_task,
                    **sigproc_v1_task,
                    **lnfit_task,
                    **classify_rf_task,
                )
                run_descs += [run_desc]

                # for classify jobs that involve PTMs or MHC, we'll do run reporting
                # differently rather than emitting a section for each run.
                if not ptm_report and not mhc_report and not pro_report:
                    self.report_section_markdown(f"# RUN {run_desc.run_name}")
                    self.report_section_run_object(run_desc)
                    if test_rf_task or nn_v1_task:
                        self.report_section_from_template(
                            "train_and_test_template.ipynb"
                        )

        self.report_section_markdown(f"# JOB {self.job}")
        self.report_section_job_object()

        if ptm_report:
            self.report_section_from_template("train_and_test_template_ptm.ipynb")
        elif mhc_report:
            self.report_section_from_template("train_and_test_template_mhc.ipynb")
        elif pro_report:
            self.report_section_from_template("train_and_test_template_pro.ipynb")
        else:
            self.report_section_from_template("train_and_test_epilog_template.ipynb")

        n_runs = len(run_descs)
        if n_runs > 1 and sigproc_tasks[0]:
            # TASK: better logic for when to include spike_template.  --spike?
            self.report_section_from_template("spike_template.ipynb")

        sigproc_imports_desc = ""
        if sigproc_tasks[0]:
            sigproc_imports_desc = "## Sigproc imports:\n"
            sigproc_imports_desc += "\n".join(
                [f"\t* {s.ims_import.inputs.src_dir}" for s in sigproc_tasks]
            )

            self.report_section_first_run_object()
            self.report_section_from_template("sigproc_v1_template.ipynb")
            self.report_section_from_template("classify_template.ipynb")

        self.report_preamble(
            utils.smart_wrap(
                f"""
                # Classify Overview
                ## {n_runs} run_desc(s) processed.
                ## Sample: {self.sample}
                ## Job: {self.job}
                This file generated by {current_file_and_line_str()}.
                {sigproc_imports_desc}
            """,
                width=None,
            )
        )

        return run_descs
