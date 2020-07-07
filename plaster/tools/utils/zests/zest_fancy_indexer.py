import numpy as np
from plumbum import local
from munch import Munch
from zest import zest
from plaster.tools.utils.fancy_indexer import FancyIndexer
from plaster.tools.log.log import debug


def zest_fancy_indexer():
    # Build a 2 dimension list of 2-D images.
    # This is (1, 2, 3, 3)
    a = np.array(
        [
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9],],
                [[10, 20, 30], [40, 50, 60], [70, 80, 90],],
            ]
        ]
    )
    lengths = a.shape[0:2]
    _fn = lambda i, j: a[i, j]

    def it_indexes_3_dims_with_sigletons():
        idx = FancyIndexer(lengths=lengths, lookup_fn=_fn)
        got = idx[0, 0]
        assert np.all(got == a[0, 0])
        got = idx[0, 1]
        assert np.all(got == a[0, 1])

    def it_indexes_3_dims_with_slices():
        idx = FancyIndexer(lengths=lengths, lookup_fn=_fn)
        got = idx[0, :]
        assert np.all(got == a[0, :])

    def it_passes_context():
        c = "some context"

        def _test(i, j, context):
            nonlocal c
            assert context == c
            return np.array([1, 2])

        idx = FancyIndexer(lengths=lengths, lookup_fn=_test, context=c)
        got = idx[0, :]
        assert np.all(got == [[1, 2], [1, 2]])

    zest()
