import math

from munch import Munch
from plaster.run.prep.prep_params import PrepParams
from plaster.tools.log import log
from zest import zest


def zest_prep_params_validate():
    def _fake_protein(abundance):
        return Munch(name="a", sequence="a", abundance=abundance)

    # Abundance data all nones
    abundance_data_all_nones = PrepParams(
        proteins=[_fake_protein(None), _fake_protein(None)]
    )

    # Normalized abundance data
    with zest.mock(log.info) as m_log:
        normalized_abundance_data = PrepParams(
            proteins=[_fake_protein(10), _fake_protein(1)]
        )
        assert m_log.not_called()  # No warning

    # Normalized abundance data with zeros
    normalized_abundance_data_with_zeros = PrepParams(
        proteins=[_fake_protein(10), _fake_protein(1), _fake_protein(0)]
    )

    # Unnormalized abundance data
    with zest.mock(log.info) as m_log:
        unnormalized_abundance_data = PrepParams(
            proteins=[_fake_protein(10), _fake_protein(5)]
        )
        assert m_log.called_once()
        assert unnormalized_abundance_data.proteins[0].abundance == 2

    # Unnormalized abundance data with zeros
    with zest.mock(log.info) as m_log:
        unnormalized_abundance_data_with_zeros = PrepParams(
            proteins=[_fake_protein(10), _fake_protein(5), _fake_protein(0)]
        )

    # Abundance data with None
    with zest.raises():
        normalized_abundance_data = PrepParams(
            proteins=[_fake_protein(10), _fake_protein(None)]
        )

    # Abundance data with nan
    with zest.raises():
        normalized_abundance_data = PrepParams(
            proteins=[_fake_protein(10), _fake_protein(math.nan)]
        )

    zest()
