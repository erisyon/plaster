from plaster.run.calib.calib import Calib
from plaster.tools.schema.schema import Params, SchemaValidationFailed
from plaster.tools.schema.schema import Schema as s
from plaster.tools.log import log
from plaster.run.sigproc_v2 import sigproc_v2_common as common
from plumbum import local
from plaster.tools.log.log import debug


class SigprocV2Params(Params):
    """
    About Calibration:
        The long term goal of the calibration files is to dissociate
        the name of the file from the records (subjects) in the file.
        For now, we're going to load all records from the calibration file
    """

    defaults = dict(
        divs=5,
        peak_mea=11,
        sig_limit=20.0,  # Keep peaks this many times brighter than the calibration background
        snr_thresh=2.0,  # Keep peaks with SNR greater than this
        n_fields_limit=None,
        run_anomaly_detection=False,
        run_regional_balance=True,
        run_analysis_gauss2_fitter=False,
        run_bandpass_filter=True,
        run_focal_adjustments=False,
        run_aligner=True,
        # TODO: Derive the follow during calibration by spectral analysis (ie, 2 std of the power spectrum)
        # ALSO: This needs to be moved into the calibration because it can not allowed to be
        # different from the calibration results because the calibration bakes in the PSF
        # as a function of these parameters.
        run_neighbor_stats=False,
        run_per_cycle_peakfinder=True,
        low_inflection=0.03,
        low_sharpness=50.0,
        high_inflection=0.50,
        high_sharpness=50.0,
        self_calib=False,
        no_calib=False,
        no_calib_psf_sigma=1.8,
        instrument_identity=None,
        locs=None,
        save_full_signal_radmat_npy=False,
    )

    schema = s(
        s.is_kws_r(
            calibration_file=s.is_str(noneable=True, required=False),
            instrument_identity=s.is_str(noneable=True),
            mode=s.is_str(options=common.SIGPROC_V2_MODES),
            divs=s.is_int(),
            peak_mea=s.is_int(),
            sig_limit=s.is_float(),
            snr_thresh=s.is_float(),
            n_fields_limit=s.is_int(noneable=True),
            run_anomaly_detection=s.is_bool(),
            run_regional_balance=s.is_bool(),
            run_analysis_gauss2_fitter=s.is_bool(),
            run_focal_adjustments=s.is_bool(),
            run_bandpass_filter=s.is_bool(),
            run_aligner=s.is_bool(),
            run_neighbor_stats=s.is_bool(),
            run_per_cycle_peakfinder=s.is_bool(),
            low_inflection=s.is_float(),
            low_sharpness=s.is_float(),
            high_inflection=s.is_float(),
            high_sharpness=s.is_float(),
            self_calib=s.is_bool(noneable=True),
            no_calib=s.is_bool(noneable=True),
            no_calib_psf_sigma=s.is_float(noneable=True),
            locs=s.is_list(elems=s.is_float(), noneable=True),
            save_full_signal_radmat_npy=s.is_bool(),
        )
    )

    def validate(self):
        # Note: does not call super because the override_nones is set to false here
        self.schema.apply_defaults(self.defaults, apply_to=self, override_nones=False)
        self.schema.validate(self, context=self.__class__.__name__)

        if self.mode == common.SIGPROC_V2_ILLUM_CALIB:
            pass
            # ZBS: At the moment these checks are more trouble than they are worth
            # if local.path(self.calibration_file).exists():
            #     if not log.confirm_yn(
            #         f"\nCalibration file '{self.calibration_file}' already exists "
            #         "when creating a SIGPROC_V2_PSF_CALIB. Overwrite?",
            #         "y",
            #     ):
            #         raise SchemaValidationFailed(
            #             f"Not overwriting calibration file '{self.calibration_file}'"
            #         )

        else:
            # Analyzing
            if self.self_calib:
                assert (
                    self.calibration_file is None
                ), "In self-calibration mode you may not specify a calibration file"
                assert (
                    self.instrument_identity is None
                ), "In self-calibration mode you may not specify an instrument identity"
                assert (
                    self.no_calib is not True
                ), "In self-calibration mode you may not specify the no_calib option"

            elif (
                not self.no_calib
                and self.calibration_file != ""
                and self.calibration_file is not None
            ):
                self.calibration = Calib.load_file(
                    self.calibration_file, self.instrument_identity
                )

            elif self.no_calib:
                assert (
                    self.no_calib_psf_sigma is not None
                ), "In no_calib mode you must specify an estimated no_calib_psf_sigma"

        return True
