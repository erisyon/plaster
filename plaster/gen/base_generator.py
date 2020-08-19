import hashlib
import itertools
import json
import re
from collections import defaultdict

import numpy as np
from munch import Munch
from plaster.gen import helpers, task_templates
from plaster.tools.log.log import debug
from plaster.tools.schema import check
from plaster.tools.schema.schema import Schema as s
from plaster.tools.schema.schema import SchemaValidationFailed
from plaster.run.sigproc_v2 import sigproc_v2_common
from plaster.tools.utils import utils
from plumbum import local


class BaseGenerator(Munch):
    """
    Base of all generators.

    Expects sub-classes to provide a class member "required_schema"
    which is used for parsing the kwargs on the __init__()
    """

    schema = None  # Should be overloaded in any sub-class
    defaults = {}  # Should be overloaded in any sub-class

    job_setup_schema = s(
        s.is_kws_r(
            job=s.is_str(help="See Main Help"),
            sample=s.is_str(allow_empty_string=False, help="See Main Help"),
        )
    )

    protein_schema = s(
        s.is_kws_r(
            protein=s.is_list(elems=s.is_kws_r(id=s.is_str(), seqstr=s.is_str(),)),
            protein_of_interest=s.is_list(
                s.is_str(allow_empty_string=False),
                noneable=True,
                help="The id of the protein(s) of interest, used in survey and reporting",
            ),
        )
    )

    label_set_schema = s(
        s.is_kws_r(label_set=s.is_list(elems=s.is_str(), help="See Main Help"))
    )

    lnfit_schema = s(
        s.is_kws_r(
            lnfit_name=s.is_list(s.is_str(), noneable=True, help="See Main Help"),
            lnfit_params=s.is_list(s.is_str(), noneable=True, help="See Main Help"),
            lnfit_dye_on_threshold=s.is_list(
                s.is_int(), noneable=True, help="See Main Help"
            ),
            lnfit_photometry_only=s.is_list(
                s.is_str(), noneable=True, help="See Main Help"
            ),
        )
    )

    scope_run_schema = s(
        s.is_kws_r(
            n_edmans=s.is_int(help="See Main Help"),
            n_pres=s.is_int(help="See Main Help"),
            n_mocks=s.is_int(help="See Main Help"),
        )
    )

    peptide_setup_schema = s(
        s.is_kws_r(
            protease=s.is_list(elems=s.is_str(), help="See Main Help"),
            decoys=s.is_str(help="See Main Help"),
            random_seed=s.is_int(noneable=True, help="See Main Help"),
            n_ptms_limit=s.is_int(
                bounds=(0, 12),
                help="Max number of PTMs per peptide to allow.  Peptides with more PTM sites than this will not consider any PTM permutations.",
            ),
        )
    )

    sim_schema = s(
        s.is_kws_r(
            n_samples_train=s.is_int(bounds=(1, None), help="See Main Help"),
            n_samples_test=s.is_int(bounds=(1, None), help="See Main Help"),
        )
    )

    classify_schema = s(
        s.is_kws_r(
            nn_v1=s.is_bool(help="Include nn_v1 classifier", noneable=True),
            nn_v2=s.is_bool(help="Include nn_v2 classifier", noneable=True),
            rf=s.is_bool(help="Include rf classifier", noneable=True),
            report_prec=s.is_list(
                elems=s.is_float(bounds=(0.001, 0.999)),
                help="The precision for classifier reporting",
            ),
        )
    )

    sigproc_source_schema = s(
        s.is_kws_r(
            sigproc_source=s.is_str(help="See Main Help"),
            movie=s.is_bool(noneable=True, help="See Main Help"),
            n_frames_limit=s.is_int(
                bounds=(1, 500), noneable=True, help="See Main Help"
            ),
        )
    )

    sigproc_v1_schema = s(
        s.is_kws_r(
            radial_filter=s.is_float(
                noneable=True, bounds=(0.01, 1.0), help="See Main Help"
            ),
            peak_find_n_cycles=s.is_int(bounds=(1, 10000), help="See Main Help"),
            peak_find_start=s.is_int(bounds=(0, 10000), help="See Main Help"),
            anomaly_iqr_cutoff=s.is_int(bounds=(1, 100), help="See Main Help"),
        )
    )

    sigproc_v2_schema = s(s.is_kws_r(calibration_file=s.is_str()))

    sigproc_v2_instrument_calib_schema = s(
        s.is_kws_r(sigproc_source=s.is_list(s.is_str(), help="See Main Help"),)
    )

    report_metadata = Munch(
        metadata=Munch(
            kernelspec=Munch(
                display_name="Python 3", language="python", name="python3"
            ),
            language_info=Munch(
                codemirror_mode=Munch(name="ipython", version=3),
                file_extension=".py",
                mimetype="text/x-python",
                name="python",
                nbconvert_exporter="python",
                pygments_lexer="ipython3",
                version="3.6.7",
            ),
        ),
        nbformat=4,
        nbformat_minor=2,
    )

    error_model_schema = s(
        s.is_kws_r(
            err_p_edman_failure=s.is_list(elems=s.is_str(help="See Main Help")),
            err_p_detach=s.is_list(elems=s.is_str(help="See Main Help")),
            err_dye_beta=s.is_list(elems=s.is_str(help="See Main Help")),
            err_dye_sigma=s.is_list(elems=s.is_str(help="See Main Help")),
            err_p_bleach_per_cycle=s.is_list(elems=s.is_str(help="See Main Help")),
            err_p_non_fluorescent=s.is_list(elems=s.is_str(help="See Main Help")),
        )
    )

    error_model_defaults = Munch(
        err_p_edman_failure=0.06,
        err_p_detach=0.05,
        err_dye_beta=7500.0,
        err_dye_sigma=0.16,
        err_dye_gain=7500.0,
        err_dye_vpd=0.1,
        err_p_bleach_per_cycle=0.05,
        err_p_non_fluorescent=0.07,
    )

    code_block = Munch(
        cell_type="code", execution_count=None, metadata=Munch(), outputs=[], source=[]
    )

    markdown_block = Munch(cell_type="markdown", metadata=Munch(), source=[])

    def __init__(self, **kwargs):
        # APPLY defaults and then ask user for any elements that are not declared

        super().__init__(**kwargs)
        self.apply_defaults()
        self.setup_err_model()
        self.validate()

        self._report_sections = []
        self._report_preamble = None
        self._validate_protein_of_interest()

    def _nb_template_path(self, template_name):
        return local.path(__file__) / "../nb_templates" / template_name

    def _validate_protein_of_interest(self):
        if "protein" in self:
            seq_ids = {seq["id"] for seq in self.protein}
            for poi in self.protein_of_interest:
                if poi not in seq_ids:
                    raise ValueError(
                        f"protein_of_interest '{poi}' is not in the protein id list. "
                        f"Confirm you specified a Name and not a UniprotAC"
                    )

    def setup_err_model(self):
        err_param_dict = defaultdict(list)
        for name, type, _, user_data in self.error_model_schema.requirements():
            values = self.get(name, [])
            for value in values:
                low_prob, high_prob, step_prob = None, None, 1

                parts = value.split("|")
                if len(parts) == 2:
                    dye_part = parts[0]
                    prob_parts = parts[1]
                else:
                    dye_part = None
                    prob_parts = parts[0]

                prob_parts = prob_parts.split(":")

                if name in ("err_p_edman_failure", "err_p_detach"):
                    if dye_part:
                        raise SchemaValidationFailed(
                            f"error model term '{name}' is not allowed to have a dye-index."
                        )
                else:
                    if dye_part is None:
                        raise SchemaValidationFailed(
                            f"error model term '{name}' expected a dye-index."
                        )

                low_prob = float(prob_parts[0])
                if len(prob_parts) > 1:
                    high_prob = float(prob_parts[1])
                if len(prob_parts) > 2:
                    step_prob = int(prob_parts[2])
                if high_prob is None:
                    high_prob = low_prob

                key = f"{name}:{dye_part if dye_part is not None else 0}"
                err_param_dict[key] += np.linspace(
                    low_prob, high_prob, step_prob
                ).tolist()
                err_param_dict[key] = list(set(err_param_dict[key]))
        self.err_param_dict = err_param_dict

    def apply_defaults(self):
        """Overloadable by sub-classes."""
        self.schema.apply_defaults(self.defaults, self, override_nones=True)

    def validate(self):
        """Overloadable by sub-classes for extra validation"""
        self.schema.validate(self, context=self.__class__.__name__)

    def ims_imports(self, sigproc_source):
        if self.movie:
            ims_import = task_templates.ims_import(
                sigproc_source, is_movie=True, n_cycles_limit=self.n_frames_limit
            )
        else:
            ims_import = task_templates.ims_import(sigproc_source, is_movie=False)

        return ims_import

    def sigprocs_v1(self):
        tasks = {}
        if self.sigproc_source:
            ims_import = self.ims_imports(self.sigproc_source)
            sigproc = task_templates.sigproc_v1()
            sigproc.sigproc_v1.parameters.radial_filter = self.radial_filter
            sigproc.sigproc_v1.parameters.peak_find_n_cycles = self.peak_find_n_cycles
            sigproc.sigproc_v1.parameters.peak_find_start = self.peak_find_start
            sigproc.sigproc_v1.parameters.anomaly_iqr_cutoff = self.anomaly_iqr_cutoff
            tasks = Munch(**ims_import, **sigproc)
        return tasks

    # def sigprocs_v2(self, **kwargs):
    #     tasks = {}
    #     if self.sigproc_source:
    #         ims_import = self.ims_imports(self.sigproc_source)
    #         sigproc = task_templates.sigproc_v2_analyze(**kwargs)
    #         tasks = Munch(**ims_import, **sigproc)
    #     return tasks

    def lnfits(self):
        # It is common to have multiple lnfit tasks for a single run, so this fn returns a
        # block with potentially multiple lnfit tasks using unique task names when more
        # than one is present.
        lnfit_tasks = {}
        if self.lnfit_params:
            if not self.lnfit_dye_on_threshold:
                raise ValueError(
                    f"You must specify a --lnfit_dye_on_threshold when --lnfit_params is given"
                )

            dye_thresholds = self.lnfit_dye_on_threshold
            lnfit_names = self.lnfit_name or ([None] * len(self.lnfit_params))
            photometries_only = self.lnfit_photometry_only or (
                [True] * len(self.lnfit_params)
            )

            if len(self.lnfit_params) > 1 and len(dye_thresholds) == 1:
                dye_thresholds *= len(self.lnfit_params)

            assert len(self.lnfit_params) == len(dye_thresholds)
            assert len(self.lnfit_params) == len(lnfit_names)

            for i, (params, thresh, name, photometry_only) in enumerate(
                zip(self.lnfit_params, dye_thresholds, lnfit_names, photometries_only)
            ):
                task = task_templates.lnfit()
                task.lnfit.parameters["lognormal_fitter_v2_params"] = params
                task.lnfit.parameters["dye_on_threshold"] = thresh
                task.lnfit.parameters["photometry_only"] = photometry_only.lower() in (
                    "true",
                    "1",
                )

                task_name = "lnfit"
                if len(self.lnfit_params) > 1 or name:
                    task_name = name or f"lnfit_{i}"
                    helpers.task_rename(task, task_name)
                lnfit_tasks[task_name] = task[task_name]
        return lnfit_tasks

    def run_name(self, aa_list, protease=None, err_set=None):
        """
        A helper for run folder names based on aa_list and protease.
        Note, not all generators will use this convention.

        Compose a run_name from protease and aa_list in normalized form:
        Eg: protease="trypsin", aa_list=("DE", "K") => "trypsin_de_k"
        """
        if protease is None:
            protease = ""
        aa_list = [a.replace("[", "").replace("]", "") for a in aa_list]
        aa = "_".join(aa_list)
        if err_set is not None:
            err_str = hashlib.md5(json.dumps(err_set).encode()).hexdigest()[0:4]
        else:
            err_str = ""
        return re.sub(
            "[^0-9a-z_]+",
            "_",
            (protease + ("_" if protease != "" else "") + aa).lower() + "_" + err_str,
        )

    def _label_str_permutate(self, label_str):
        """
        Return list of permutations of a label_str such as:
        "A,B,C:2" => ("A", "B"), ("A", "C"), ("B", "C")

        A suffix label set may be added to each permutation with +:
        "A,B,C:2+S" => ("A", "B", "S"), ("A", "C", "S"), ("B", "C", "S")
        "A,B,C:2+S,T" => ("A", "B", "S", "T"), ("A", "C", "S", "T"), ("B", "C", "S", "T")
        """

        check.t(label_str, str)
        semi_split = label_str.split(":")

        if len(semi_split) > 2:
            raise ValueError(f"Label-set '{label_str}' has >1 colon.")

        suffix_labels = ""
        if len(semi_split) == 2:
            suffix_split = semi_split[1].split("+")

            if len(suffix_split) > 2:
                raise ValueError(f"Label-set '{label_str}' has >1 plus.")

            if len(suffix_split) == 2:
                semi_split = [semi_split[0], suffix_split[0]]
                suffix_labels = suffix_split[1].split(",")
                suffix_labels = [slabel.strip() for slabel in suffix_labels]

        labels = semi_split[0].split(",")
        labels = [label.strip() for label in labels]

        if len(semi_split) == 1:
            perm_count = len(labels)
        else:
            perm_count = int(semi_split[1])
            if not 0 < perm_count < len(labels):
                raise ValueError(
                    f"Label-set '{label_str}' has a permutation count "
                    f"of {perm_count}; needs to be between 0 and {len(labels) - 1}"
                )

        perms = list(itertools.combinations(labels, perm_count))

        if suffix_labels:
            perms = [p + tuple(suffix_labels) for p in perms]

        return perms

    def label_set_permutate(self):
        check.list_t(self.label_set, str)
        return utils.flatten(
            [self._label_str_permutate(label_str) for label_str in self.label_set], 1
        )

    def error_set_permutate(self):
        tuples = [
            [(key, val) for val in vals] for key, vals in self.err_param_dict.items()
        ]
        return tuples

    def run_parameter_permutator(self):
        """
        Generate permutations of all the variable parameters
        Defaults all arguments to self.*
        Gracefully handles lack of protease.
        """
        proteases = utils.non_none(self.get("protease"), [None])
        if len(proteases) == 0:
            proteases = [None]
        proteases = [("protease", p) for p in proteases]

        label_sets = self.label_set_permutate()
        label_sets = [("label_set", s) for s in label_sets]

        err_sets = self.error_set_permutate()

        combined = [proteases, label_sets] + err_sets

        for params in itertools.product(*combined):
            protease = utils.filt_first(params, lambda i: i[0] == "protease")
            protease = protease[1]
            label_set = utils.filt_first(params, lambda i: i[0] == "label_set")
            label_set = label_set[1]

            # Given that the label_set is now known, the error model can be setup
            n_channels = len(label_set)
            err_set = Munch(
                p_edman_failure=[self.error_model_defaults.err_p_edman_failure] * 1,
                p_detach=[self.error_model_defaults.err_p_detach] * 1,
                dye_beta=[self.error_model_defaults.err_dye_beta] * n_channels,
                dye_sigma=[self.error_model_defaults.err_dye_sigma] * n_channels,
                dye_gain=[self.error_model_defaults.err_dye_gain] * n_channels,
                dye_vpd=[self.error_model_defaults.err_dye_vpd] * n_channels,
                p_bleach_per_cycle=[self.error_model_defaults.err_p_bleach_per_cycle]
                * n_channels,
                p_non_fluorescent=[self.error_model_defaults.err_p_non_fluorescent]
                * n_channels,
            )

            for param in params:
                if param[0].startswith("err_"):
                    parts = param[0].split(":")
                    err_set[parts[0][4:]][int(parts[1])] = param[
                        1
                    ]  # The 4: removes the "err_"

            yield protease, label_set, err_set

    def erisyon_block(self, aa_list, protease=None, err_set=None):
        return task_templates.erisyon(
            run_name=self.run_name(aa_list, protease, err_set),
            sample=self.sample,
            generator_name=self.__class__.__name__,
        )

    def _markdown_to_markdown_block(self, markdown):
        lines = [f"{line}\n" for line in markdown.split("\n")]
        block = Munch(**self.markdown_block)
        block.source = lines
        return block

    def report_preamble(self, markdown):
        """A a preamble in markdown format"""
        self._report_preamble = markdown

    def report_section_markdown(self, markdown):
        self._report_sections += [("markdown", markdown)]

    def report_section_run_object(self, run):
        self._report_sections += [
            ("code", [f'run = RunResult("./{run.run_name}")'],),
        ]

    def report_section_first_run_object(self):
        self._report_sections += [
            ("code", [f"run = job.runs[0]"],),
        ]

    def report_section_job_object(self):
        self._report_sections += [
            ("code", [f'job = JobResult(".")'],),
        ]

    def report_section_user_config(self):
        """
        Emit report configuation parameters specified by the user via gen so that they
        can be further edited if desired, and used by reporting functions in the templates.
        """
        config = []
        if self.protein_of_interest:
            config += [f"PGEN_protein_of_interest = {self.protein_of_interest}\n"]
        if self.report_prec:
            config += [f"PGEN_report_precisions = {self.report_prec}\n"]
        if config:
            self.report_section_markdown("# PGEN-controlled report config")
            config = [
                f"# These values were or can be specified by the user at gen time:\n"
            ] + config
            self._report_sections += [("code", config)]

    def report_section_run_array(self, runs, to_load=None):
        to_load_string = "" if to_load is None else f", to_load={to_load}"
        run_names = [run.run_name for run in runs]
        self._report_sections += [
            (
                "code",
                [
                    f"run_names = {run_names}\n"
                    f'runs = [RunLoader(f"./{{name}}"{to_load_string}) for name in run_names]'
                ],
            )
        ]

    def report_section_from_template(self, template_name):
        """Write the report from its pieces"""
        self._report_sections += [("template", template_name)]

    def report_assemble(self):
        """Assemble the report from its pieces. A giant Munch is returned"""
        report = Munch(**self.report_metadata)
        report.cells = []

        preamble_block = self._markdown_to_markdown_block(self._report_preamble)
        report.cells += [preamble_block]

        # LOAD all templates
        templates_by_name = {}
        for section_type, section_data in self._report_sections:
            if section_type == "template":
                templates_by_name[section_data] = utils.json_load_munch(
                    self._nb_template_path(section_data)
                )

        # FIND all of the @IMPORT-MERGE blocks
        import_merge = []
        for _, template in templates_by_name.items():
            for cell in template.cells:
                if cell.cell_type == "code":
                    first_line = utils.safe_list_get(cell.source, 0, "")
                    if "# @IMPORT-MERGE" in first_line:
                        for line in cell.source:
                            if "import" in line:
                                import_merge += [line]

        import_merge += ["from plaster.tools.zplots import zplots\n"]
        import_merge = sorted(list(set(import_merge))) + ["z=zplots.setup()"]
        import_block = Munch(**self.code_block)
        import_block.source = import_merge
        report.cells += [import_block]

        for section_type, section_data in self._report_sections:
            if section_type == "code":
                lines = section_data
                block = Munch(**self.code_block)
                block.source = lines
                report.cells += [block]

            elif section_type == "markdown":
                block = self._markdown_to_markdown_block(section_data)
                report.cells += [block]

            elif section_type == "template":
                file_path = section_data
                template = templates_by_name[file_path]
                for cell in template.cells:
                    if cell.cell_type == "code":
                        first_line = utils.safe_list_get(cell.source, 0, "")

                        if (
                            "@IMPORT-MERGE" not in first_line
                            and "@REMOVE-FROM-TEMPLATE" not in first_line
                        ):
                            block = Munch(**self.code_block)
                            block.source = cell.source
                            report.cells += [block]

                    if cell.cell_type == "markdown":
                        block = Munch(**self.markdown_block)
                        block.source = cell.source
                        report.cells += [block]

        return report

    def report_task(self):
        pass

    def generate(self):
        """
        Abstract method to be overloaded.
        Expected to return a list of runs.
        """
        pass
