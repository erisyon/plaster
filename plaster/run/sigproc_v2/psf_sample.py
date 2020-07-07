import numpy as np
import psf
from plaster.tools.log.log import debug


def psf_sample(n_z_slices=64, depth_in_microns=0.4, r_in_microns=28.0):
    assert n_z_slices > 1  # psf.PSF fails if shape == 1
    args = dict(
        shape=(n_z_slices, 128),  # number of samples in z and r direction
        dims=(
            depth_in_microns,
            r_in_microns,
        ),  # size in z and r direction in micrometers
        ex_wavelen=640.0,  # excitation wavelength in nanometers
        em_wavelen=665.0,  # emission wavelength in nanometers
        num_aperture=1.49,
        refr_index=1.51,
        magnification=1.0,
        pinhole_radius=1.50,  # in micrometers
        pinhole_shape="round",
    )
    obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
    width = 8
    width2 = width // 2
    psfs = []
    for zi in range(n_z_slices):
        im = psf.mirror_symmetry(obsvol.empsf.slice(zi))
        mea = im.shape[0] + 1
        mea2 = mea // 2
        im = im[mea2 - width2 : mea2 + width2 - 1, mea2 - width2 : mea2 + width2 - 1]
        im = im / np.sum(im)
        psfs += [im]
    return np.array(psfs)
