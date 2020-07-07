import numpy as np
import math
from munch import Munch
from scipy.spatial import distance
from plaster.tools.image import imops
from plaster.tools.image.coord import XY, YX, WH, HW, ROI
from plaster.tools.log.log import debug


class Synth:
    """
    Generate synthetic images for a single field
    Results
        self.unit_peak
        self.peak_dim
        self.peak_xs
        self.peak_ys
        self.peak_locs
        self.closest
        self.peak_areas
        self.frame_xs
        self.frame_ys
        self.ims
        self.peak_focus
    """

    def __init__(
        self,
        n_peaks=1,  # Number of peaks to add
        n_cycles=1,  # Number of frames to make in the stack
        n_channels=1,  # Number of channels.
        dim=(512, 512),  # dims of frames
        bg_bias=0,  # bias of the background
        bg_std=0.5,  # std of the background noise per channel
        peak_focus=1.0,  # The std of the peaks (all will be the same)
        peak_mean=(40.0,),  # Mean of the peak area distribution for each channel
        peak_std=2.0,  # Std of the peak areas distribution
        peak_xs=None,  # Used to put peaks at know locations
        peak_ys=None,  # Used to put peaks at know locations
        peak_area_all_cycles=None,  # Areas for all peaks on each cycle (len == n_cycles)
        peak_area_by_channel_cycle=None,  # Areas, all peaks, channels, cycles (len ==  n_channels * n_cycles * n_peaks)
        grid_distribution=False,  # When True the peaks are laid out in a grid
        all_aligned=False,  # When True all cycles will be aligned
        digitize=False,
        frame_offsets=None,
        anomalies=None,  # int, number of anomalies per image
        random_seed=None,
    ):

        if random_seed is not None:
            np.random.seed(random_seed)

        if not isinstance(bg_bias, tuple):
            bg_bias = tuple([bg_bias] * n_channels)

        if not isinstance(bg_std, tuple):
            bg_std = tuple([bg_std] * n_channels)

        self.bg_bias = bg_bias
        assert len(bg_bias) == n_channels

        self.bg_std = bg_std
        assert len(bg_std) == n_channels

        self.dim = HW(dim)

        self.anomalies = []

        # Peaks are a floating point positions (not perfectly aligned with pixels)
        # and the Gaussians are sampled around those points

        self.n_peaks = n_peaks
        self.n_cycles = n_cycles
        self.n_channels = n_channels
        self.peak_mean = peak_mean
        self.peak_std = peak_std
        self.peak_focus = peak_focus

        # unit_peak is only a reference; the real peaks are sub-pixel sampled but will have the same dimensions
        self.unit_peak = imops.generate_gauss_kernel(self.peak_focus)

        self.peak_dim = self.unit_peak.shape[0]

        if peak_xs is not None and peak_ys is not None:
            # Place peaks at specific locations
            self.peak_xs = np.array(peak_xs)
            self.peak_ys = np.array(peak_ys)
        else:
            # Place peaks in patterns or random
            if grid_distribution:
                # Put the peaks on a grid
                n_cols = int(math.sqrt(n_peaks) + 0.5)
                ixs = np.remainder(np.arange(n_peaks), n_cols)
                iys = np.arange(n_peaks) // n_cols
                border = 15
                self.peak_xs = (self.dim.w - border * 2) * ixs / n_cols + border
                self.peak_ys = (self.dim.h - border * 2) * iys / n_cols + border
            else:
                # Distribute the peaks randomly
                self.peak_xs = np.random.uniform(
                    low=self.peak_dim + 1.0,
                    high=self.dim.w - self.peak_dim - 1.0,
                    size=n_peaks,
                )
                self.peak_ys = np.random.uniform(
                    low=self.peak_dim + 1.0,
                    high=self.dim.h - self.peak_dim - 1.0,
                    size=n_peaks,
                )

        self.peak_locs = [XY(x, y) for x, y in zip(self.peak_xs, self.peak_ys)]
        if self.n_peaks > 0:
            peak_dists = distance.cdist(self.peak_locs, self.peak_locs, "euclidean")
            peak_dists[peak_dists == 0] = 10000.0
            self.closest = peak_dists.min(axis=0)

        self.peak_areas = np.empty((n_channels, n_cycles, n_peaks))
        if peak_area_all_cycles is not None:
            # Use one area for all peaks, all channels for each cycle
            assert len(peak_area_all_cycles) == n_cycles
            for cycle in range(n_cycles):
                self.peak_areas[:, cycle, :] = peak_area_all_cycles[cycle]
        elif peak_area_by_channel_cycle is not None:
            # Specified areas for each peak, each channel, each cycle
            assert peak_area_by_channel_cycle.shape == (n_channels, n_cycles, n_peaks)
            self.peak_areas[:] = peak_area_by_channel_cycle[:]
        elif n_peaks > 0:
            # Make random peak areas by channel means
            for channel in range(n_channels):
                self.peak_areas[channel, :, :] = np.random.normal(
                    loc=self.peak_mean[channel],
                    scale=self.peak_std,
                    size=(n_cycles, n_peaks),
                )
                self.peak_areas[channel] = np.clip(
                    self.peak_areas[channel], 0.0, 1000.0
                )

        # Frames are integer aligned because this is the best that the aligner would be able to do
        if frame_offsets is None:
            self.frame_xs = np.random.randint(low=-5, high=5, size=n_cycles)
            self.frame_ys = np.random.randint(low=-5, high=5, size=n_cycles)
        else:
            self.frame_xs = [i[1] for i in frame_offsets]
            self.frame_ys = [i[0] for i in frame_offsets]

        # Cycle 0 always has no offset
        self.frame_xs[0] = 0
        self.frame_ys[0] = 0

        if all_aligned:
            self.frame_xs = np.zeros((n_cycles,))
            self.frame_ys = np.zeros((n_cycles,))

        self.ims = np.zeros((n_channels, n_cycles, dim[0], dim[1]))
        for cycle in range(n_cycles):
            for channel in range(n_channels):
                # Background has bg_std plus bias
                im = bg_bias[channel] + np.random.normal(
                    size=dim, scale=self.bg_std[channel]
                )

                # Peaks are on the pixel lattice from their floating point positions
                # No signal-proportional noise is added
                for x, y, a in zip(
                    self.peak_xs, self.peak_ys, self.peak_areas[channel, cycle, :]
                ):
                    # The center of the peak is at the floor pixel and the offset is the fractional part
                    _x = self.frame_xs[cycle] + x
                    _y = self.frame_ys[cycle] + y
                    ix = int(_x + 0.5)
                    iy = int(_y + 0.5)
                    frac_x = _x - ix
                    frac_y = _y - iy
                    g = imops.generate_gauss_kernel(
                        self.peak_focus, offset_x=frac_x, offset_y=frac_y
                    )
                    imops.accum_inplace(im, g * a, loc=XY(ix, iy), center=True)

                # overwrite with random anomalies if specified
                if anomalies is not None:
                    if cycle == 0:
                        # in cycle 0 pick location and size of anomalies for this image
                        anomalies_for_channel = []
                        self.anomalies += [anomalies_for_channel]
                        for i in range(anomalies):
                            sz = np.random.randint(10, 100, size=2)
                            l = np.random.randint(0, self.dim[0], size=2)
                            anomalies_for_channel += [(l, sz)]
                    for l, sz in self.anomalies[channel]:
                        # in all cycles, vary the location, size, and intensity of the anomaly,
                        # and print it to the image.
                        l = l * (1.0 + np.random.random() / 10.0)
                        sz = sz * (1.0 + np.random.random() / 10.0)
                        im[ROI(l, sz)] = im[ROI(l, sz)] * np.random.randint(2, 20)

                if digitize:
                    im = (im.clip(min=0) + 0.5).astype(np.uint8)  # +0.5 to round up
                else:
                    im = im.clip(min=0)
                self.ims[channel, cycle, :, :] = im

    def field_stack(self):
        return self.ims
