import math

from munch import Munch
from plaster.tools.log import log
from plaster.tools.schema.schema import Params
from plaster.tools.schema.schema import Schema as s
from plaster.tools.schema.schema import SchemaValidationFailed


class PrepParams(Params):
    ALLOW_NONES_AND_NANS_IN_ABUNDANCE = False
    NORMALIZE_ABUNDANCE = True

    defaults = Munch(
        protease=None,
        decoy_mode=None,
        include_misses=0,
        n_peps_limit=None,
        drop_duplicates=False,
        n_ptms_limit=None,
    )

    schema = s(
        s.is_kws_r(
            protease=s.is_list(noneable=True, elems=s.is_str()),
            decoy_mode=s.is_str(noneable=True),
            include_misses=s.is_int(),
            n_peps_limit=s.is_int(noneable=True),
            drop_duplicates=s.is_bool(),
            n_ptms_limit=s.is_int(noneable=True),
            proteins=s.is_list(
                s.is_kws(
                    name=s.is_str(required=True),
                    sequence=s.is_str(required=True),
                    ptm_locs=s.is_str(noneable=True),
                    report=s.is_int(noneable=True),
                    abundance=s.is_number(noneable=True),
                )
            ),
        )
    )

    def validate(self):
        super().validate()

        # Try to normalize abundance values if provided. If abundance values are provided, do basic validation.
        # If no abundance values are provided, do nothing.
        # When a protein csv with no abundance columns is provided, it will come through as all nans

        abundance_info_present = any(
            hasattr(protein, "abundance")
            and protein.abundance is not None
            and not math.isnan(protein.abundance)
            for protein in self.proteins
        )

        if abundance_info_present:
            abundance_criteria = [
                (lambda protein: hasattr(protein, "abundance"), "Abundance missing"),
                (
                    lambda protein: protein.abundance >= 0
                    if protein.abundance is not None
                    else True,
                    "Abundance must be greater than or equal to zero",
                ),
            ]

            if not self.ALLOW_NONES_AND_NANS_IN_ABUNDANCE:
                abundance_criteria += [
                    (
                        lambda protein: protein.abundance is not None,
                        "Abundance must not be None",
                    ),
                    (
                        lambda protein: not math.isnan(protein.abundance),
                        "Abundance must not be NaN",
                    ),
                ]

            # Find min abundance value, also check for zeros and NaNs and error if found
            min_abundance = None
            for protein in self.proteins:
                # Check to make sure abundance passes criteria
                for criteria_fn, msg in abundance_criteria:
                    if not criteria_fn(protein):
                        abundance_value = getattr(protein, "abundance")
                        raise SchemaValidationFailed(
                            f"Protein {protein.name} has invalid abundance: {abundance_value} - {msg}"
                        )

                # Find min abundance value
                if min_abundance is None or (
                    protein.abundance < min_abundance and protein.abundance > 0
                ):
                    min_abundance = protein.abundance

            if self.NORMALIZE_ABUNDANCE:
                if min_abundance != 1:
                    log.info("abundance data is not normalized, normalizing.")
                    # normalize abundance by min value
                    for protein in self.proteins:
                        if protein.abundance is not None:
                            protein.abundance /= min_abundance
        else:
            # Abundance information is missing from all proteins
            # Set abudance to 1
            for protein in self.proteins:
                protein.abundance = 1
