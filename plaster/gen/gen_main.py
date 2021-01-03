#!/usr/bin/env python

import re
import sys
import tempfile
import time
import uuid

from munch import Munch
from plaster.gen import helpers
from plaster.gen.classify_v1_generator import ClassifyV1Generator
from plaster.gen.classify_v2_generator import ClassifyV2Generator
from plaster.gen.errors import ValidationError
from plaster.gen.ptm_generator import PTMGenerator
from plaster.gen.sigproc_v1_generator import SigprocV1Generator
from plaster.gen.sigproc_v2_calib_generator import SigprocV2CalibGenerator
from plaster.gen.sigproc_v2_generator import SigprocV2Generator
from plaster.gen.survey_generator import SurveyGenerator
from plaster.run.run import RunExecutor
from plaster.tools.aaseq.proteolyze import proteases as protease_dict
from plaster.tools.assets import assets
from plaster.tools.log.log import (
    confirm_yn,
    debug,
    error,
    important,
    info,
    input_request,
)
from plaster.tools.schema.schema import SchemaValidationFailed
from plaster.tools.uniprot import uniprot
from plaster.tools.utils import data, tmp, utils
from plumbum import FG, cli, colors, local

"""

Plaster Generator (gen) architecture

Gen consists of one plumbum cli application and many "generators".
Each run of gen invokes one generator.

The Generators provide:
    * A schema.
    * A generate method which generates the jobs & runs & reports.

The main of the Gen app:
    * Creates the requested generator
    * Assembles a variety of switches into a state generator request.
    * The generator request fields might no have been specified on the CLI
      and, if allowed, the GenApp will ask the user.
    * Some of those switches are global switches that are available to
      all generators; others are added dynamically upon the request of the
      "cli" field in the userdata of the generator's scehema.
    * The generator request is passed as kwargs into the
      constructor of the Generator; the constructor then validates.
    * The generator then is asked to "generate()"

Notes
    There's tension in this design. One the one hand the generators
    feel like Plumbum's sub-commands. But the vast majority of
    the switches are tools switches. If I treat each generator
    as a subcommand then plumbums rules mean that you'd have to organize
    the calls like:
        gen --protein=XXX --protein=YYY ptm --ptm_specific_switch

    Which really is weird because the "ptm" statement belongs first.

    So to resolve that I bypass plumbum's subcommand concept and I
    dynamically inject switches into the Gen App before it is instanciated.

"""


VERSION = "0.2"


# The following help has magic markup that is parsed in the help()
def help_template(generators):
    return utils.smart_wrap(
        f"""
        PGEN -- The plaster run generator

        VERSION: {VERSION}
        TASK: License, version, etc.

        Usage
        ------------------------------------------------------------------------------
        gen <GENERATOR> <SWITCHES>

        Example Usage:
        --------------
            gen classify_v2 \\
                --protein_uniprot=P01308 \\
                --n_edmans=10 \\
                --label_set='DE,C' \\
                --job=example \\
                --sample=insulin

        #SWITCHES
        ===============================================================================
        @--job='./my_run'                             # (See: GENERATORS)
        @--sample='a modified protein'                # (See: GENERATORS)

        Protein import (All are Repeatable; 1+ Required)...
        ------------------------------------------------------------------------------
        @--protein_fasta='local.fasta'                # Local file (See: FASTA)
        ^--protein_fasta='http://a.com/a.fasta'       # URL of same
        @--protein_csv='//jobs_folder/local.csv'     # Local-file (See: CSV)
        @--protein_csv='http://a.com/a.csv'           # URL of same
        @--protein_csv='s3://bucket/folder/a.csv'     # S3 source of same
        @--protein_seq='Insulin:MALWMRLLPLL'          # Sequence in-line (See SEQS)
        @--protein_uniprot='P01308'                   # Lookup by Uniprot AC
        @--protein_uniprot='Insulin:P01308'           # Lookup AC and change name
        ^

        Protein options (All are Repeatable; All Optional)...
        ------------------------------------------------------------------------------
        @--protein_random=N                           # Of proteins added, pick N
        @--protein_of_interest='P10636-8'             # Affects classify reporting
        ^--protein_exclude='MyName2'                  # Exclude name
        ^--protein_abundance='P37840-1:10000.0'       # Specify abundance by name
        ^--protein_abundance-csv='some.csv'           # Specify abundance (See: CSV)
        ^--protein_abundance-csv='http://a.com/a.csv' # URL of same


        Common Generator Switches: (See: GENERATORS)...
        ------------------------------------------------------------------------------
        @--label_set='DE,Y,C,K:2+S'                   # Repeatable (See: LABELS)
        @--protease='trypsin'                         # Repeatable (See: PROTEASES)
        @--scheme='trypsin/DE,Y,C,K:2+S'              # Repeatable (See: SCHEME)
        @--n_edmans=10                                # Edman cycles (See: LABELS)
        @--n_pres=1                                   # default: 1 (See: LABELS)
        @--n_mocks=0                                  # default: 0 (See: LABELS)
        @--decoys='reverse'                           # default: None. See (DECOYS)
        @--random_seed=123                            # default: None
        @--report_prec=.9                             # classifier precision to report

        Error Model: (See: ERROR_MODEL)...
        ------------------------------------------------------------------------------
        @--err_p_edman_failure=0.06                   # Edman miss
        @--err_p_detach=0.05                          # Surface detach
        @--err_row_k_beta=1.0                         # Mean of row adjustment
        @--err_row_k_sigma=0.16                       # Stdev. of row adjustment

                                                      # The following probabilities
                                                      # are specified per-dye like:
                                                      # "dye|prob" where dye count
                                                      # starts at zero.
        @--err_dye_beta=0|7500                        # Brightness per dye
        @--err_dye_sigma=0|0.16                       # Log-normal stdev
        @--err_dye_zero_beta=0|0                      # Brightness of the "zero count"
        @--err_dye_zero_sigma=0|200                   # Std of zero count
        @--err_p_bleach_per_cycle=0|0.05              # Bleach rate
        @--err_p_non_fluorescent=0|0.07               # Dud rate

        ^                                             # The following probabilities
        ^                                             # are specified per-aa-label
        ^                                             # like: "aa:prob" where aa
        ^                                             # matches a --label_set
        ^--err_p_failure_to_bind_amino_acid=0.0       # Failure to bind AA
        ^--err_p_failure_to_attach_to_dye=0.0         # Failure to attach to dye


        Sigproc Setup (Optional)...
        ------------------------------------------------------------------------------
        @--sigproc_source='s3://bucket/folder'        # S3 source (See: SIGPROC)
        ^--sigproc_source='http://a.com/a'            # URL of same
        ^--sigproc_source='./folder'                  # Local path of same
        @--anomaly_iqr_cutoff                         # [0,100] default: 95
        @--lnfit_name                                 # Repeatable (See: LNFIT)
        @--lnfit_params                               # Repeatable (See: LNFIT)
        @--lnfit_dye_on_threshold                     # Repeatable (See: LNFIT)
        @--peak_find_n_cycles                         # [1,ncycles] default: 4
        @--peak_find_start                            # [0,ncycles-1] default: 0
        @--radial_filter                              # [0,1.0] or default: None

        Less-frequently used switches...
        ------------------------------------------------------------------------------
        @--cache_folder='...'                         # default:
                                                      # $ERISYON_TMP/gen_cache
        @--force                                      # Force clean
        @--overwrite                                  # Force overwrite (danger)
        @--run_name='a'                               # Force run name (danger)
        @--prop='a.b=1=int'                           # Modify a property (danger)
        @--skip_report                                # Do not gen. report

        #GENERATORS & JOBS & SAMPLES
        ===============================================================================
        Generators are a mode under which this script creates job
        instructions.  All executions of this script require a generator
        be specified in the first argument.

        Generators emit "JOBS" into Job folders as named with the --job=
        switch into the ./jobs_folder folder. Note that ./jobs_folder might be
        a sym-link to somewhere else.

        Current generators are:
            {colors.yellow|generators}

        Each Generator may require specific switches which may be
        enumerate with "gen <GENNAME> --help"

        When a Generator is not given a required input, it will ask for it manually.

        Generators may choose to emit more than one RUN into the job folder
        in which case there may be more than on sub-folder of the job.

        A sample is a required human-readable string that describes the
        biological sample this came from.

        #ERROR_MODEL
        ===============================================================================
        All of the error model probabilities can be swept in the form:
          --err_p_bleach_per_cycle=0|0.05:0.07:3

        which means "The probability of bleach per cycle for dye 0 shall
        be swept from 0.05 to 0.07 in 3 steps.

        Note that for --err_p_edman_failure and --err_p_detach
        you do not prefix with a "dye:". Example "--err_p_detach=0.01:0.02:3"

        Be careful when you use the iterators as the number of permutations
        can grow extremely quickly and thus generate a very large number of runs.

        #URLs
        ===============================================================================
        Any switch which accepts a file will also accept an http, https, or s3 URL.

        #FASTA
        ===============================================================================
        .fasta files should be in the Uniprot form.
        See https://www.uniprot.org/help/fasta-headers

        #CSV
        ===============================================================================
        .csv files require a mandatory single line header as follows in any order:
            Name, Seq, Abundance, UniprotAC, PTM, POI

        If UniprotAC is given the Seq will be filled from the UniprotAC.
        If UniprotAC is given but Name isn't, it will use the AC as the Name.
        Abundance is optional. In the case that the abundance alone is given
        then it can be used to assign abundances
        to proteins that were imported in the --protein_* commands.
        PTM is optional.  It is a semi-colon-separate list of 1-based aa-locations
        at which PTM can be performed (e.g. phosphorylation).
        POI is optional and contains a 0 or a 1. Used to specify "proteins of interest"

        Quoted and un-quoted fields are legal and columns are separated by commas.

        #SEQS
        ===============================================================================
        Protein and peptide sequences are specified in IUPAC; N to C order.
        (http://publications.iupac.org/pac/1984/pdf/5605x0595.pdf)

        Special rules:
            * Whitespace is ignored
                "AB CD" = "ABCD"
            * "." can be used in place of "X"
                "AB..CD" = "ABXXCD"
            * Anything wrapped in () is dropped.
                "AB(a comment)CD" = "ABCD"
            * Square brackets are modifications of the previous amino-acid,
              usually used to indicate a Post-Translational-Modification (PTM)
                "AS[p]D" = "A" + "S[p]" + "D"
            * Curly brackets are reserved for future use

        #LABELS
        ===============================================================================
        Examples:
          "C,K"           = Label C in channel 0, K in ch. 1.
          "DE,C,K"        = Label D and E in channel 0, C in ch. 1, K in ch. 2.
          "DE,C,K: 2"     = Choose all 2 label permutations, eg: (DE,C) (DE,K) (C,K)
          "DE,C,K: 2+S,T" = Choose all 2 label permutations and add label(s)
                            e.g. (DE,C,S,T) (DE,K,S,T) (C,K,S,T)
          "DE,C[p]"       = Label D and E in channel 0, and phospho C in ch. 1.

        Peptides are degraded by Edman degradation one amino acid at at time
        from the N-terminus. When a labelled amino-acid is cleaved the loss in
        fluorescence is what guides identification. The --n_edmans=X parameter
        specifies the number of Edman cycles. More cycles will sequence deeper into
        the peptides but also adds more time to the experiment.

        #PROTEASES
        ===============================================================================
        Proteolyze the proteins and any decoys with one or more of:
            {colors.yellow|", ".join(list(protease_dict.keys())[0:5])}
            {colors.yellow|", ".join(list(protease_dict.keys())[5:])}

        You may also proteolyze with more than one protease simultaneously using the
        syntax e.g. --protease=lysc+endopro

        #SCHEME
        ===============================================================================
        A scheme is a combination of protease and label_set. Use the scheme argument to
        specify schemes directly. The format is protease/label_set.

        #DECOYS
        ===============================================================================
        Decoys are protein sequences which are expected to *not* be present
        in a sample and are used to estimate the "False Discovery Rate"
        (ie. the rate at which the classifier makes incorrect calls.)
        In cases where decoys are helpful, this option will generate decoys
        automatically.
        Option available for decoy are:
            "none", "reverse", "shuffle"
        These options are applied before proteolysis.

        #SIGPROC
        ===============================================================================
        When one or more --sigproc_source= are given, the data from an instrument
        run will be added into the analysis.

        #LNFIT
        ===============================================================================
        When one or more --lnfit_params are given, lnfit tasks will be executed on each
        sigproc_source dataset.  The --lnfit_params string specified will be passed
        directly to the pflib lnfit routine.

        The --lnfit_dye_on_threshold parameter is used to convert sigproc intensities
        at each cycle to the "ON/OFF" track_photometries.csv input format required by
        pflib's lnfit routine.  An intensity above this threshold is considered "ON".

        You may specifiy a single --lnfit_dye_on_threshold to be used for all lnfit
        tasks, or specifiy a separate threshold for each.

        --lnfit_name may optionally be specified for each parameter set to assign
        a user-specified folder name for the lnfit task.  Otherwise, the tasks will
        be auto-numbered in the case there is more than one, e.g. lnfit_0, lnfit_1...

        Examples:
            --lnfit_name=lnfit_647_t4000_b7000_p1r10a95
            --lnfit_params='-c 1 -w 647 -m 4 -o 0 -e 10 -s HLAA --beta 7000 --truncate 2'
            --lnfit_dye_on_threshold=4000

        """,
        assert_if_exceeds_width=True,
    )


def add_switches_to_cli_application_from_schema(app, schema, reserved_field_names=[]):
    """
    Add plumbum switches into app from schema top-level fields.
    Arguments:
        app: cli.Application. to add switches into
        schema: Schema. From which switches will be created.
        reserved_field_names: List[str].
            Any field in the schema that is in the reserved_field_names is skipped;
            those are handled by special-case code. This applies to --protein and
            other switches that are derived by more complicated assembolies.
    """
    fields = schema.top_level_fields()
    for field_name, field_type, field_help, field_userdata, field_subtype in fields:
        if field_name not in reserved_field_names:
            is_list = field_type is list
            if is_list:
                field_type = field_subtype

            is_bool = field_type is bool
            if is_bool:
                switch = cli.Flag([f"--{field_name}"], help=field_help, default=None)
            else:
                switch = cli.SwitchAttr(
                    [f"--{field_name}"], field_type, help=field_help, list=is_list
                )
            setattr(app, field_name, switch)


class GenFuncs:
    def _request_field_from_user(self, field_name, type_, default):
        """Mock point"""
        headless = ValueError(f"Attempt to request field {field_name} in headless mode")

        while True:
            resp = input_request(
                f"Enter {field_name} ({type_.__name__} default={default}): ",
                default_when_headless=headless,
            )
            try:
                if resp == "":
                    resp = default
                if resp is None:
                    val = None
                else:
                    val = type_(resp)
            except Exception:
                important(f"Unable to convert '{resp}' to {type_}. Try again.")
            else:
                break

        return val

    def _write_runs(self, job_folder, run_descs, props=[]):
        """
        Convert the munch run_descs into folders
        """

        if not job_folder.exists():
            job_folder.mkdir()

        found_run_names = {}

        for i, run in enumerate(run_descs):
            # FIND or OVERRIDE run_name
            run_name = run.get("run_name")

            assert run_name not in found_run_names
            found_run_names[run_name] = True

            # SETUP _erisyon block
            if "_erisyon" not in run:
                run._erisyon = Munch()
            run._erisyon.run_i = i
            run._erisyon.run_i_of = len(run_descs)
            run._erisyon.run_name = run_name

            # OVERRIDE with props
            for prop in props:
                k, v, t = prop.split("=")
                if t == "bool":
                    v = True if v == "true" else False
                elif t == "int":
                    v = int(v)
                elif t == "float":
                    v = float(v)
                elif t == "int_list":
                    v = [int(i) for i in v.split(",")]
                elif t == "int_dict":
                    v = v.split(",")
                    v = {v[i]: int(v[i + 1]) for i in range(0, len(v), 2)}
                else:
                    raise TypeError(f"Unknown type in prop coversion '{t}")
                utils.block_update(run, k, v)

            # Keep the run_name out
            run.pop("run_name", None)
            folder = job_folder / run_name
            folder.mkdir()
            RunExecutor(folder, tasks=run).save()

            info(f"Wrote run to {folder}")


class GenApp(cli.Application, GenFuncs):
    PROGNAME = colors.green | "gen"
    VERSION = VERSION
    COLOR_GROUPS = {"Switches": colors.yellow}
    DESCRIPTION = colors.green | "Generate plaster run instructions"

    # derived_vals are switch-like elements that are assembled from more-primitive switches
    derived_vals = Munch(protein=[])

    # Global switches that are always available in all generators
    # -------------------------------------------------------------------------------------
    job = cli.SwitchAttr(["--job"], str, help="Path to job folder")

    sample = cli.SwitchAttr(
        ["--sample"], str, help="Human-readable description of the biological sample"
    )

    cache_folder = cli.SwitchAttr(
        ["--cache_folder"],
        str,
        default=str(tmp.cache_folder()),
        help="Where to cache files",
    )

    force = cli.Flag(["--force"], default=False, help="Force deletion of existing job")

    overwrite = cli.Flag(
        ["--overwrite"], default=False, help="Overwrite into existing job, (dangerous)"
    )

    prop = cli.SwitchAttr(["--prop"], str, list=True, help="Set a property (dangerous)")

    run_name = cli.SwitchAttr(["--run_name"], str, help="Set run_name (dangerous)")

    protein_random = cli.SwitchAttr(
        ["--protein_random"], int, help="Pick N random proteins from set"
    )

    skip_report = cli.Flag(
        ["--skip_report"], default=False, help="Skip report generation"
    )

    symlink_to_cache = cli.Flag(
        ["--symlink_to_cache"],
        default=False,
        help="Symlink instead of copying from cache",
    )

    generator_klass_by_name = Munch(
        classify_v1=ClassifyV1Generator,
        classify_v2=ClassifyV2Generator,
        ptm=PTMGenerator,
        sigproc_v1=SigprocV1Generator,
        sigproc_v2=SigprocV2Generator,
        sigproc_v2_calib=SigprocV2CalibGenerator,
        survey=SurveyGenerator,
    )

    def _print(self, line):
        """Mock-point"""
        print(line)

    @cli.switch(["--readme"], help="A comprehensive guide for this tool")
    def readme(self):
        """
        Parse the help file
        ^ = Feature not yet implemented
        @ = Feature implemented
        # = Header line

        @--switch='value:1'     # comment
        """
        switch_pat = re.compile(
            r"""
            ^(?P<leading_char>[\^\@\#]?)
            (?P<switch>[^= ]+)?
            (?P<value>=\S*)?
            (?P<comment>.*\#.*)?
            """,
            re.VERBOSE,
        )

        generators = ", ".join(self._subcommands.keys())
        for line in help_template(generators).split("\n"):
            m = switch_pat.match(line)
            if m:
                if m.group("leading_char") == "^":
                    continue

                elif m.group("leading_char") == "@":
                    self._print(
                        f"{colors.yellow & colors.bold | m.group('switch') or ''}"
                        f"{m.group('value') or ''}{colors.yellow | m.group('comment') or ''}"
                    )
                    continue

                elif m.group("leading_char") == "#":
                    self._print(colors.blue & colors.bold | m.group("switch"))
                    continue

            self._print(line)
        sys.exit(0)

    @cli.switch(["--protein_seq"], str, list=True)
    def protein_seq(self, seqs):
        """
        Include protein(s) in csv format (Must have a header row with 'Name', 'Seq' and optional
        'Abundance' columns.). May be a local File-path or URL.
        """
        for seq in seqs:
            parts = seq.split(":")
            if len(parts) != 2:
                raise ValidationError(
                    f"--protein_seq arguments must be in form 'id:XXXXX' but found '{seq}'"
                )
            self.derived_vals.protein += [dict(seqstr=parts[1], id=parts[0])]

    @cli.switch(["--protein_csv"], str, list=True)
    def protein_csv(self, sources):
        """
        Include protein(s) in csv format (Must have a header row with 'Name', 'Seq' and optional
        'Abundance', 'UniprotAC', and 'PTM' columns.). May be a local File-path or URL.
        """
        for source in sources:
            source = helpers.cache_source(
                self.cache_folder, source, self.local_sources_tmp_folder
            )
            proteins_df = helpers.protein_csv_df(source)
            self.derived_vals.protein += proteins_df.to_dict("records")

    @cli.switch(["--protein_uniprot"], str, list=True)
    def protein_uniprot(self, uniprot_acs):
        """
        Include protein by uniprot AC (will fetch from uniprot.org)
        """
        for uniprot_ac in uniprot_acs:
            pro_dict = helpers.split_protein_name(uniprot_ac)
            # This is a little ugly because the get_ac call caches to the
            # default erisyon caching folder (see utils.cache) and therefore
            # bypasses --cache_folder option. This will need a re-work
            # perhaps a "with tmp.cache_folder()" type operation.
            source = uniprot.get_ac_fasta(pro_dict["seqstr"])
            self.derived_vals.protein += helpers.protein_fasta(source, pro_dict["id"])

    @cli.switch(["--protein_fasta"], str, list=True)
    def protein_fasta(self, file_paths):
        for file_path in file_paths:
            source = helpers.cache_source(
                self.cache_folder, file_path, self.local_sources_tmp_folder
            )
            self.derived_vals.protein += helpers.protein_fasta(source)

    @classmethod
    def run(cls, argv=None, exit=True):
        """
        ZBS: Plumbum subcommand startup sequence is complicated.
        But, during the default run() it instantiates this class and passes
        only the next argument which prevents me from jamming dynamic switches
        into the class. So here I duplicate the argument I need argv[1]
        into the head of the list.  And then later I have to overload _parse_args()
        in order to pop those arguments back off.

        Also, if you pass in "--help" that would normally be handled by
        plumbum correctly, but these hacks prevent that so I have
        to keep track of the construct_fail and let it proceed so that
        an instance it correctly allocated because the "help" commands
        only work on a functional instance (ie you can not raise the Help
        exception during construction).
        """
        # files spec'd to gen will be copied here for this job, and moved to
        # the job folder if the generator succeeds.
        cls.job_uuid = uuid.uuid4().hex
        cls.local_sources_tmp_folder = local.path(tempfile.gettempdir()) / cls.job_uuid
        cls.local_sources_tmp_folder.mkdir()
        cls.construct_fail = False

        if not argv or len(argv) < 2 or argv[1].startswith("--"):
            if argv is not None and argv[1] == "--readme":
                # This is a crazy work-around to get the app instance
                # to construct so I can print the readme.
                cls.construct_fail = True
                inst = super(GenApp, cls).run(
                    argv=["", "calib", "--job=foo"], exit=False
                )
                inst[0].readme()
                return 0

            cls.construct_fail = True
            error(
                "You must specify a generator as the first argument after 'gen'.\n"
                f"Options are {', '.join(GenApp.generator_klass_by_name.keys())}"
            )
            argv = ["gen", "--help"]

        if argv is not None:
            return super(GenApp, cls).run(
                argv=[utils.safe_list_get(argv, 1)] + argv, exit=exit
            )
        else:
            return super(GenApp, cls).run(argv=argv, exit=exit)

    def validate_job_name_and_folder(self):
        """
        Validate the job name and compute job_folder path.
        Optionally delete the job_folder if it exists.

        Returns:
             job_folder path
        """

        self.job = assets.validate_job_folder(self.job)
        delete_job = False
        if self.overwrite:
            delete_job = False
        elif self.force:
            delete_job = True
        elif self.job.exists():
            delete_job = confirm_yn(
                (
                    colors.red & colors.bold
                    | f"Do you really want to remove ALL contents of "
                )
                + (
                    colors.yellow
                    | f"'{self.job}'?\nIf no, then job may be in an inconsistent state.\n"
                ),
                "y",
            )

        if delete_job:
            important(f"Deleting all of {self.job}.")
            self.job.delete()

        return self.job

    def _parse_args(self, argv):
        """See above for why this crazy code pops. Undoing the list munging."""
        if self.construct_fail:
            return super()._parse_args(argv)

        argv.pop(0)
        argv.pop(0)
        return super()._parse_args(argv)

    def _validate_args(self, swfuncs, tailargs):
        """See above for why this is overloaded."""
        if self.construct_fail:
            tailargs = []
        return super()._validate_args(swfuncs, tailargs)

    def _cache_s3_reference(self, source):
        if source.startswith("s3:"):
            found_cache, cache_path = tmp.cache_path("plaster_s3", source)
            if not found_cache:
                important(f"Syncing from {source} to {cache_path}")
                local["aws"]["s3", "sync", source, cache_path] & FG

            # A little bit of a hack to apply this to an url but it does what we want
            source_folder_name = local.path(source).basename

            if self.symlink_to_cache:
                local["ln"][
                    "-s",
                    cache_path,
                    self.local_sources_tmp_folder / source_folder_name,
                ]()
            else:
                local["cp"][
                    "-r",
                    cache_path,
                    self.local_sources_tmp_folder / source_folder_name,
                ]()

            # Hardcoding relative path here. It will always be the same 3 levels: run/plaster_output/task
            return "../../../_gen_sources/" + source_folder_name

        return source

    def __init__(self, generator_name):
        if self.construct_fail:
            return super().__init__(generator_name)

        klass = GenApp.generator_klass_by_name.get(generator_name)
        self.generator_klass = klass
        if klass is None:
            raise ValidationError(
                f"Unknown generator '{generator_name}'. Options are: "
                f"{', '.join(list(GenApp.generator_klass_by_name.keys()))}"
            )

        # Dynamically create plumbum switches based on the generator
        add_switches_to_cli_application_from_schema(
            GenApp, klass.schema, reserved_field_names=[self.derived_vals.keys()]
        )

        super().__init__(generator_name)

    def main(self):
        if self.construct_fail:
            return

        job_folder = self.validate_job_name_and_folder()

        schema = self.generator_klass.schema
        defaults = self.generator_klass.defaults

        requirements = schema.requirements()
        # APPLY defaults and then ask user for any elements that are not declared
        generator_args = {}
        switches = self._switches_by_name

        if self.protein_random is not None:
            info(f"Sampling {self.protein_random} random proteins from imported set")
            n = len(self.derived_vals.protein)
            assert n >= self.protein_random
            self.derived_vals.protein = data.subsample(
                self.derived_vals.protein, self.protein_random
            )
            assert len(self.derived_vals.protein) == self.protein_random

        for arg_name, arg_type, _, arg_userdata in requirements:
            if (
                arg_name in self.derived_vals
                and self.derived_vals.get(arg_name) is not None
            ):
                # Load from a derived switch (eg: protein)
                generator_args[arg_name] = self.derived_vals[arg_name]
            elif arg_name in switches and switches.get(arg_name) is not None:
                # Load from a switch
                generator_args[arg_name] = getattr(self, arg_name)
            else:
                # If the schema allows the user to enter manually
                if arg_userdata.get("allowed_to_be_entered_manually"):
                    generator_args[arg_name] = self._request_field_from_user(
                        arg_name, arg_type, default=defaults.get(arg_name)
                    )

        # Download sigproc sources and replace with local path before handing to generator
        if "sigproc_source" in generator_args:
            source = generator_args["sigproc_source"]
            if source is not None and source.startswith("s3:"):
                generator_args["sigproc_source"] = self._cache_s3_reference(source)

        # Intentionally run the generate before the job folder is written
        # so that if generate fails it doesn't leave around a partial job.
        gen_sources_folder = job_folder / "_gen_sources"
        try:
            generator_args["force_run_name"] = self.run_name
            generator_args["local_sources_tmp_folder"] = self.local_sources_tmp_folder
            generator_args["gen_sources_folder"] = gen_sources_folder
            generator = self.generator_klass(**generator_args)
            run_descs = generator.generate()
        except (SchemaValidationFailed, ValidationError) as e:
            # Emit clean failure and exit 1
            error(str(e))
            return 1

        # WRITE the job & copy any file sources
        self._write_runs(job_folder, run_descs, props=self.prop)
        gen_sources_folder.delete()
        self.local_sources_tmp_folder.move(gen_sources_folder)

        if not self.skip_report:
            for report_name, report_builder in generator.reports.items():
                report = report_builder.report_assemble()
                if report is not None:
                    utils.json_save(job_folder / f"{report_name}.ipynb", report)

        utils.yaml_write(
            job_folder / "job_manifest.yaml",
            uuid=self.job_uuid,
            localtime=time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime()),
            # Note: it seems localtime inside our container is UTC
            who=local.env.get("RUN_USER", "Unknown"),
            cmdline_args=sys.argv,
        )


if __name__ == "__main__":
    # This is ONLY executed if you do not "main.py" because
    # main.py imports this file as a subcommand
    GenApp.run()
