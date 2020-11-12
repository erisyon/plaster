import math

from munch import Munch
from plaster.run.prep.prep_params import PrepParams
from plaster.tools.log import log
from zest import zest


def zest_prep_params_validate():
    def _fake_protein(abundance):
        return Munch(name="a", sequence="a", abundance=abundance)

    def it_allows_all_nones_in_abundance_data():
        abundance_data_all_nones = PrepParams(
            proteins=[_fake_protein(None), _fake_protein(None)]
        )

    def it_allows_missing_abundance_data():
        abundance_data_missing_abundance = PrepParams(
            proteins=[Munch(name="a", sequence="a"), Munch(name="a", sequence="a")]
        )
        assert all(p.abundance == 1 for p in abundance_data_missing_abundance.proteins)

    def it_allows_all_nans():
        """
        This is the case that occurs when a protein csv is provided with no abundance column
        """
        abundance_data_missing_abundance = PrepParams(
            proteins=[_fake_protein(math.nan), _fake_protein(math.nan)]
        )
        assert all(p.abundance == 1 for p in abundance_data_missing_abundance.proteins)

    def it_doesnt_warn_when_abundance_data_is_already_normalized():
        with zest.mock(log.info) as m_log:
            normalized_abundance_data = PrepParams(
                proteins=[_fake_protein(10), _fake_protein(1)]
            )
            assert m_log.not_called()  # No warning

    def it_allows_zeros_in_abundance_data():
        normalized_abundance_data_with_zeros = PrepParams(
            proteins=[_fake_protein(10), _fake_protein(1), _fake_protein(0)]
        )

    def it_warns_and_normalized_unnormalized_abundance_data():
        with zest.mock(log.info) as m_log:
            unnormalized_abundance_data = PrepParams(
                proteins=[_fake_protein(10), _fake_protein(5)]
            )
            assert m_log.called_once()
            assert unnormalized_abundance_data.proteins[0].abundance == 2

    def it_allows_zeros_in_unnormalized_abundance_data():
        with zest.mock(log.info) as m_log:
            unnormalized_abundance_data_with_zeros = PrepParams(
                proteins=[_fake_protein(10), _fake_protein(5), _fake_protein(0)]
            )

    def it_doesnt_allow_none_in_abundance_data():
        with zest.raises():
            normalized_abundance_data = PrepParams(
                proteins=[_fake_protein(10), _fake_protein(None)]
            )

    def it_doesnt_allow_nan_in_abundance_data():
        with zest.raises():
            normalized_abundance_data = PrepParams(
                proteins=[_fake_protein(10), _fake_protein(math.nan)]
            )

    zest()
