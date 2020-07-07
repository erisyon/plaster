import numpy as np
from zest import zest
from plaster.run.train_rf import train_rf_worker as worker
from plaster.tools.log.log import debug


def zest_train_rf():
    def it_subsamples():
        """
        It should be able to extract n_subsample rows (2)
        from each peptide where which peptide si originally
        sampled at n_samples (8) rows.
        """

        n_samples = 8
        n_peptides = 5
        n_cycles = 3

        y0 = np.repeat(np.arange(n_peptides), (n_samples,))
        X0 = np.random.uniform(size=(n_samples * n_peptides, n_cycles))

        n_subsample = 2
        X1, y1 = worker._subsample(n_subsample, X0, y0)
        assert np.all(y1 == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

        assert X1.shape == (n_subsample * n_peptides, n_cycles)

        _X0 = np.sum(X0, axis=1)
        _X1 = np.sum(X1, axis=1)

        assert np.all(np.isin(_X1[0], _X0[0:n_samples]))
        assert np.all(np.isin(_X1[1], _X0[0:n_samples]))

        for i in range(_X1.shape[0]):
            source_pep = i // n_subsample
            assert np.isin(
                _X1[i], _X0[source_pep * n_samples : (source_pep + 1) * n_samples],
            )

    zest()
