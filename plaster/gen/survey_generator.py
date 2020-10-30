from munch import Munch
from plaster.gen import task_templates
from plaster.gen.base_generator import BaseGenerator
from plaster.tools.log.log import debug, important
from plaster.tools.schema.schema import Schema as s
from plaster.tools.utils import utils


class SurveyGenerator(BaseGenerator):
    """
    Survey protease-label schemes to identify best candidates for a
    given protein/peptides domain.
    """

    schema = s(
        s.is_kws_r(
            **BaseGenerator.job_setup_schema.schema(),
            **BaseGenerator.protein_schema.schema(),
            **BaseGenerator.label_set_schema.schema(),
            **BaseGenerator.peptide_setup_schema.schema(),
            **BaseGenerator.error_model_schema.schema(),
            **BaseGenerator.scope_run_schema.schema(),
            **BaseGenerator.scheme_schema.schema(),
        )
    )

    #

    defaults = Munch(
        n_pres=1,
        n_mocks=0,
        n_edmans=15,
        protein_of_interest=None,
        decoys="none",
        random_seed=None,
        n_ptms_limit=0,
    )

    error_model_defaults = Munch(
        err_p_edman_failure=0.0,
        err_p_detach=0.0,
        err_dye_beta=7500.0,
        err_dye_sigma=0.16,
        err_dye_zero_beta=0.0,
        err_dye_zero_sigma=200.0,
        err_p_bleach_per_cycle=0.0,
        err_p_non_fluorescent=0.0,
    )

    def generate(self):
        # To start we model the survey task structure identically on how normal
        # sim/classify runs are done -- one run per protease/label-scheme.
        # A single job-level report then gathers the results from the runs
        # and presents a table indicating the predicted best schemes for the
        # objective function of interest (settable in the report itself).

        # TODO: This could be made much faster by forgoing the simplicity of
        # the task/run structure noted above, and instead creating a single
        # "run" that processes all permutations.  Further, a version of sim,
        # or mods to sim, could be made to eliminiate some steps that are
        # not necessary when all we really require is the 'perfect' dyetracks
        # for the peptides.  We currently achieve this by simply setting
        # n_samples to 1 to cause the least possible amount of simulation,
        # and setting all error-model probability params to 0.

        run_descs = []
        for protease, aa_list, err_set in self.run_parameter_permutator():

            prep_task = task_templates.prep(
                self.protein,
                protease,
                self.decoys,
                proteins_of_interest=self.protein_of_interest,
                n_ptms_limit=self.n_ptms_limit,
            )

            sim_v2_task = task_templates.sim_v2(
                list(aa_list),
                err_set,
                n_pres=self.n_pres,
                n_mocks=self.n_mocks,
                n_edmans=self.n_edmans,
                n_samples_train=1,
                n_samples_test=1,
                is_survey=True,
            )
            sim_v2_task.sim_v2.parameters.random_seed = self.random_seed
            # note: same seed is used to generate decoys

            survey_v2_task = task_templates.survey_v2()

            e_block = self.erisyon_block(aa_list, protease, err_set)

            run_name = f"{e_block._erisyon.run_name}"
            if self.force_run_name is not None:
                run_name = self.force_run_name

            run_desc = Munch(
                run_name=run_name,
                **e_block,
                **prep_task,
                **sim_v2_task,
                **survey_v2_task,
            )
            run_descs += [run_desc]

        self.report_section_job_object()
        self.report_section_from_template("survey_template.ipynb")
        self.report_preamble(
            utils.smart_wrap(
                f"""
                # NNSurvey Overview
                ## {len(run_descs)} run_desc(s) processed.
                ## Sample: {self.sample}
                ## Job: {self.job}
            """
            )
        )

        return run_descs
