import numpy as np
from zest import zest
from plaster.run.calib.calib import Calib, RegIllum, RegPSF, CalibIdentity


def zest_reg_illum():
    def it_sets():
        reg_illum = RegIllum(2, 512, 5)

        vals0 = np.random.uniform(size=(5, 5))
        reg_illum.set(0, vals0)

        vals1 = np.random.uniform(size=(5, 5))
        reg_illum.set(1, vals1)

        assert np.all(vals0 == reg_illum.reg_illum[0])
        assert np.all(vals1 == reg_illum.reg_illum[1])

    def it_interpolates():
        reg_illum = RegIllum(1, 512, 5)
        vals = np.ones((5, 5))
        vals[0, 0] = 0.5
        reg_illum.set(0, vals)
        im = reg_illum.interp(0)
        assert im.shape == (512, 512)
        assert np.allclose(im[0, 0], 0.5)
        assert np.allclose(im[511, 511], 1.0)

    zest()


def zest_reg_psf():
    raise NotImplementedError
    zest()


def zest_calib():
    def it_adds_reg_illum():
        reg_illum = RegIllum(2, 512, 5)
        calib = Calib()
        calib.add_reg_illum(reg_illum, CalibIdentity("foo"))
        test_path = "/tmp/_test.calib"
        calib.save_file(test_path)
        calib = Calib.load_file(test_path, CalibIdentity("foo"))
        assert np.all(calib.reg_illum().reg_illum == reg_illum.reg_illum)

    def it_adds_reg_psf():
        reg_psf = RegPSF.fixture(n_channels=2)
        calib = Calib()
        calib.add_reg_psf(reg_psf, CalibIdentity("foo"))
        test_path = "/tmp/_test.calib"
        calib.save_file(test_path)
        calib = Calib.load_file(test_path, CalibIdentity("foo"))
        assert np.all(calib.reg_psf().params == reg_psf.params)

    def adds_without_ident():
        reg_psf = RegPSF.fixture(n_channels=2)
        test_path = "/tmp/_test.calib"

        def it_enforces_identity():
            calib = Calib()
            calib.add_reg_psf(reg_psf)
            with zest.raises(ValueError, in_args="calib identity not specified"):
                calib.save_file(test_path)

        def it_rewrites_identity():
            calib = Calib()
            calib.add_reg_psf(reg_psf)
            calib.set_identity(CalibIdentity("me"))
            calib.save_file(test_path)
            _calib = Calib.load_file(test_path, CalibIdentity("me"))
            assert np.all(_calib.reg_psf().params == reg_psf.params)

        zest()

    zest()
