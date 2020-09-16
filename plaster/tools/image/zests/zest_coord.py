import time
import sys
from zest import zest
from plaster.tools.image.coord import XY, YX, WH, HW, ROI, roi_center, roi_shift
import numpy as np
from plaster.tools.log.log import debug


def zest_CoordHelpers():
    def it_accepts_xy_but_stores_reversed():
        loc = XY(2, 3)
        assert loc.x == 2 and loc.y == 3 and loc == (3, 2)

    def it_accepts_yx():
        loc = YX(3, 2)
        assert loc.x == 2 and loc.y == 3 and loc == (3, 2)

    def it_accepts_wh_but_stores_reversed():
        dim = WH(2, 3)
        assert dim.x == 2 and dim.y == 3 and dim == (3, 2)

    def it_accepts_hw():
        dim = HW(3, 2)
        assert dim.x == 2 and dim.y == 3 and dim == (3, 2)

    def it_has_x_y_w_h_properties():
        loc = XY(1, 2)
        assert loc.x == 1 and loc.y == 2 and loc.w == 1 and loc.h == 2

    def it_can_add_and_sub():
        assert WH(1, 2) + WH(3, 4) == WH(4, 6) and WH(1, 2) - WH(3, 4) == WH(-2, -2)

    def it_can_floor_div_and_true_div():
        assert WH(1, 2) // 2 == WH(0, 1) and WH(1, 2) / 2 == WH(0.5, 1)

    def it_can_scalar_mul():
        assert WH(1, 2) * 2 == WH(2, 4)

    def it_floors():
        assert WH(1.1, 2.9) == WH(1, 2)

    def it_can_can_slice_and_dice_with_roi():
        dim = WH(10, 10)
        image = np.zeros(dim)
        roi = ROI(XY(1, 1), dim - WH(2, 2))
        cropped_image = image[roi]
        assert np.array_equal(image[1:9, 1:9], cropped_image[:, :])

    def it_can_can_slice_an_roi_with_centering():
        roi = ROI(XY(1, 2), WH(2, 4), center=True)
        assert roi[0] == slice(0, 4) and roi[1] == slice(0, 2)

    def it_centers_an_roi_with_a_coord():
        orig_dim = WH(100, 50)
        roi = roi_center(orig_dim, percent=0.5)
        assert roi == ROI(loc=XY(25, 12), dim=WH(50, 25))

    def it_centers_an_roi_with_a_tuple():
        orig_dim = (50, 100)  # Note this is in H, W
        roi = roi_center(orig_dim, percent=0.5)
        expected = ROI(loc=XY(25, 12), dim=WH(50, 25))
        assert roi == expected

    def it_computes_mag():
        assert XY(3, 4).mag() == 25.0

    def it_shifts_an_roi():
        roi = ROI(loc=YX(5, 10), dim=HW(15, 20))
        new_roi = roi_shift(roi, YX(-2, -3))
        assert new_roi == ROI(YX(5 - 2, 10 - 3), HW(15, 20))

    zest()
