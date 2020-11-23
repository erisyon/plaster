import plaster.tools.image.coord
from zest import zest
from plaster import tools
from plaster.tools.image.coord import XY, YX, WH, HW, ROI, roi_center
from plaster.tools.image import imops
from plaster.run.sigproc_v2 import synth
import numpy as np
from scipy import ndimage as ndi
from plaster.tools.log.log import debug


def spotty_images():
    # CREATE a test spot
    spot = imops.generate_gauss_kernel(2.0)
    spot = spot / np.max(spot)

    dim = WH(50, 50)
    spot_locs = [XY(15, 15), XY(10, 20), XY(20, 21)]

    # CREATE test images with spots
    test_images = []
    for loc in spot_locs:
        im = np.zeros(dim)
        # im = np.random.normal(0, 0.1, dim)
        # im = np.ones(dim) * 0.1
        imops.accum_inplace(im, spot, loc=loc, center=True)
        test_images += [im]

    return spot_locs, np.array(test_images)


def zest_ImageOps_clipping():
    """
    Src extends to both sides of the target
          TTTTTTTTTTTTTTTT
      SSSSSSSSSSSSSSSSSSSSSSSSSS
    """

    def it_handles_all_sliding_clips():
        # Slide a 2 wide src over a 3 wide target
        for tar_x in range(-3, 5):
            tar_l, src_l, src_w = tools.image.coord.clip1d(tar_x, tar_w=3, src_w=2)
            if tar_x <= -2 or tar_x >= 3:
                assert tar_l is None and src_l is None and src_w is None
            if tar_x == -1:
                assert tar_l == 0 and src_l == 1 and src_w == 1
            if tar_x == 0:
                assert tar_l == 0 and src_l == 0 and src_w == 2
            if tar_x == 1:
                assert tar_l == 1 and src_l == 0 and src_w == 2
            if tar_x == 2:
                assert tar_l == 2 and src_l == 0 and src_w == 1

    def it_clips_left():
        """
            TTTTTTTTTTTTTTTT
        SSSSSSSSS
        """
        tar_l, src_l, src_w = tools.image.coord.clip1d(-5, 15, 10)
        assert tar_l == 0 and src_l == 5 and src_w == 5

    def it_clips_left_totally():
        """
                   TTTTTTTTTTTTTTTT
        SSSSSSSSS
        """
        tar_l, src_l, src_w = tools.image.coord.clip1d(-10, 15, 10)
        assert tar_l is None and src_l is None and src_w is None

    def it_no_ops():
        """
        TTTTTTTTTTTTTTTT
           SSSSSSSSS
        """
        tar_l, src_l, src_w = tools.image.coord.clip1d(1, 15, 10)
        assert tar_l == 1 and src_l == 0 and src_w == 10

    def it_clips_right():
        """
        TTTTTTTTTTTTTTTT
                   SSSSSSSSS
        """
        tar_l, src_l, src_w = tools.image.coord.clip1d(12, 15, 10)
        assert tar_l == 12 and src_l == 0 and src_w == 3

    def it_clips_right_totally():
        """
        TTTTTTTTTTTTTTTT
                          SSSSSSSSS
        """
        tar_l, src_l, src_w = tools.image.coord.clip1d(15, 15, 10)
        assert tar_l is None and src_l is None and src_w is None

    def it_clips_both_sides():
        """
           TTTTTTTTT
        SSSSSSSSSSSSSS
        """
        tar_l, src_l, src_w = tools.image.coord.clip1d(-5, 10, 20)
        assert tar_l == 0 and src_l == 5 and src_w == 10

    def it_clips2d_inclusive():
        """
        ssssssssssss
        sttttttttsss
        sttttttttsss
        ssssssssssss
        """
        tar_roi, src_roi = tools.image.coord.clip2d(-1, 5, 10, -2, 4, 15)
        assert tar_roi == ROI(XY(0, 0), WH(5, 4))
        assert src_roi == ROI(XY(1, 2), WH(5, 4))

    def it_clips2d_exclusive():
        """
        sss
        sss
           ttt
           ttt
        """
        tar_roi, src_roi = tools.image.coord.clip2d(3, 3, 3, 2, 2, 2)
        assert tar_roi is None
        assert src_roi is None

    def it_clips2d_exclusive_2():
        """
            sss
            sss

           ttttt
           ttttt
           ttttt
        """
        # tar_x, tar_w, src_w, tar_y, tar_h, src_h
        # tar_roi, src_roi = imops.clip2d(97, 512, 7, -103, 512, 7)
        tar_roi, src_roi = tools.image.coord.clip2d(1, 5, 3, -3, 3, 2)
        assert tar_roi is None
        assert src_roi is None

    zest()


def zest_ImageOps():
    im0, im1, im2 = None, None, None

    def _before():
        nonlocal im0, im1, im2
        im0 = np.array([[1, 2], [3, 4]])
        im1 = np.array([[5, 6], [7, 8]])
        im2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def it_makes_a_gaussian_kernel_that_is_odd_sqaure_and_integral_one():
        for std in np.linspace(1.0, 3.0, 5):
            g = imops.generate_gauss_kernel(std)
            assert g.shape[0] & 1 == 1
            assert g.shape[0] == g.shape[1]
            assert np.abs(1.0 - g.sum()) < 0.1

    def it_makes_a_gaussian_kernel_that_has_fractaional_position():
        g = imops.generate_gauss_kernel(1.5, 0.5, 0.0, mea=9)
        com = ndi.measurements.center_of_mass(g)
        x_shift = com[1] - g.shape[1] // 2
        y_shift = com[0] - g.shape[0] // 2
        assert x_shift > 0.45 and x_shift < 0.55
        assert y_shift >= -0.01 and y_shift < 0.01

    def it_adds_a_sub_image_into_a_target():
        dst = np.ones(WH(4, 4))
        src = np.ones(WH(2, 2))
        imops.accum_inplace(dst, src, XY(1, 1))
        good = np.array([[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]])
        assert (dst == good).all()

    def it_adds_a_sub_image_into_a_target_with_clipping():
        dst = np.ones(WH(2, 2))
        src = np.ones(WH(4, 4))
        imops.accum_inplace(dst, src, XY(-1, -1))
        good = np.array([[2, 2], [2, 2]])
        assert (dst == good).all()

    def it_aligns_two_images():
        spot_locs, test_images = spotty_images()
        found_offsets, _ = imops.align(test_images)
        actual_offset = spot_locs[1] - spot_locs[0]
        assert np.all(found_offsets[1] == actual_offset)

    def it_crops():
        src = np.array([[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]])
        inner = imops.crop(src, XY(1, 1), WH(2, 2))
        assert np.array_equal(inner, np.array([[2, 2], [2, 2]]))

    def it_unravels_to_compute_a_dot_product():
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        assert imops.dot(a, b) == 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8

    def it_generates_a_circle_mask_with_even_radius():
        circle = imops.generate_circle_mask(2)
        T = True
        F = False
        expected = np.array(
            [
                [F, T, T, T, F],
                [T, T, T, T, T],
                [T, T, T, T, T],
                [T, T, T, T, T],
                [F, T, T, T, F],
            ]
        )
        assert all(circle.flatten() == expected.flatten())

    def it_generates_a_circle_mask_with_odd_radius():
        circle = imops.generate_circle_mask(3)
        T = True
        F = False
        expected = np.array(
            [
                [F, F, T, T, T, F, F],
                [F, T, T, T, T, T, F],
                [T, T, T, T, T, T, T],
                [T, T, T, T, T, T, T],
                [T, T, T, T, T, T, T],
                [F, T, T, T, T, T, F],
                [F, F, T, T, T, F, F],
            ]
        )
        assert all(circle.flatten() == expected.flatten())

    def it_generates_a_circle_mask_with_even_radius_embedded_in_larger_dim():
        circle = imops.generate_circle_mask(2, 7)
        T = True
        F = False
        expected = np.array(
            [
                [F, F, F, F, F, F, F],
                [F, F, T, T, T, F, F],
                [F, T, T, T, T, T, F],
                [F, T, T, T, T, T, F],
                [F, T, T, T, T, T, F],
                [F, F, T, T, T, F, F],
                [F, F, F, F, F, F, F],
            ]
        )
        assert all(circle.flatten() == expected.flatten())

    def it_generates_a_circle_mask_with_odd_radius_embedded_in_larger_dim():
        circle = imops.generate_circle_mask(3, 9)
        T = True
        F = False
        expected = np.array(
            [
                [F, F, F, F, F, F, F, F, F],
                [F, F, F, T, T, T, F, F, F],
                [F, F, T, T, T, T, T, F, F],
                [F, T, T, T, T, T, T, T, F],
                [F, T, T, T, T, T, T, T, F],
                [F, T, T, T, T, T, T, T, F],
                [F, F, T, T, T, T, T, F, F],
                [F, F, F, T, T, T, F, F, F],
                [F, F, F, F, F, F, F, F, F],
            ]
        )
        assert all(circle.flatten() == expected.flatten())

    def it_extracts_a_trace():
        im_times_10 = im2 * 10
        trace = imops.extract_trace(
            [im2, im_times_10], loc=XY(1, 0), dim=WH(2, 2), center=False
        )
        expected = [[[2, 3], [5, 6]], [[20, 30], [50, 60]]]
        assert np.array_equal(trace, expected)

    def it_stacks_a_list():
        s = imops.stack([im0, im1])
        assert s.ndim == 3 and np.array_equal(s, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    def it_stacks_a_singleton():
        s = imops.stack(im0)
        assert s.ndim == 3 and s.sum() == 10

    def it_no_ops_a_stack():
        s0 = imops.stack([im0, im1])
        s1 = imops.stack(s0)
        assert np.array_equal(s0, s1)

    def test_is_sets_with_a_mask():
        mask = np.array([[True, False], [False, True]], dtype=bool)
        imops.set_with_mask_in_place(im2, mask, 0, loc=XY(1, 1))
        expected = np.array([[1, 2, 3], [4, 0, 6], [7, 8, 0]])
        assert np.array_equal(im2, expected)

    def it_extracts_with_mask():
        mask = np.array([[True, False], [False, True]], dtype=bool)
        found = imops.extract_with_mask(im2, mask, XY(1, 1), center=False)
        expected = np.array([[5, 0], [0, 9]])
        assert np.array_equal(found, expected)

    def it_fills():
        im = np.ones((3, 3))
        im = imops.edge_fill(im, 1)
        # fmt: off
        assert im.tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0]
        ]
        # fmt: on

        dst = np.ones(WH(4, 4))
        imops.fill(dst, loc=XY(1, 1), dim=WH(10, 10))
        good = np.array([[1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
        assert (dst == good).all()

    def it_shifts_with_equal_ndims():
        src = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dst = imops.shift(src, YX(1, -1))
        assert np.all(dst == [[0, 0, 0], [2, 3, 0], [5, 6, 0]])

    def it_shifts_with_extra_dims():
        src = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            ]
        )
        dst = imops.shift(src, YX(1, -1))
        assert np.all(
            dst
            == [
                [[0, 0, 0], [2, 3, 0], [5, 6, 0]],
                [[0, 0, 0], [20, 30, 0], [50, 60, 0]],
            ]
        )

    zest()


def zest_ImageOps_dump():
    imstack = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    m_imstack_write = zest.stack_mock(imops._imstack_write, reset_before_each=True)

    def it_dumps_to_the_root_path_if_set():
        imops.dump_set_root_path("/dir")
        imops.dump("name", imstack)
        assert m_imstack_write.called_once_with("/dir", "name.npy", imstack)

    def it_dumps_to_local_path_if_root_not_set():
        imops.dump_set_root_path(None)
        imops.dump("name", imstack)
        assert m_imstack_write.called_once_with(None, "name.npy", imstack)

    def it_converts_lists_to_array_stacks():
        list_stack = [imstack[:, :, 0], imstack[:, :, 1]]
        imops.dump("name", list_stack)

    def it_converts_2d_arrays_to_3d():
        im = imstack[:, :, 0]
        assert im.ndim == 2
        imops.dump("name", im)

    def it_raises_on_malformed_arrays():
        with zest.raises(AssertionError):
            im = imstack[:, 0, 0]
            imops.dump("name", im)

    zest()


def zest_ImageOps_composite():
    spot_locs, test_images = spotty_images()
    first_loc = spot_locs[0]
    offsets_relative_to_first = [(loc - first_loc) for loc in spot_locs]
    comp, _ = imops.composite(test_images, offsets_relative_to_first)

    def it_composites_the_stack():
        # If it worked then there all the test peaks will have been aligned
        # causing a single peak of approx >= 3.0
        assert np.abs(3.0 - np.max(comp)) < 0.1

    def it_enlarged_the_canvas_to_maximum_offset():
        hw = HW(comp.shape)
        assert (
            hw.w == 62 and hw.h == 62
        )  # 62 because the max offset is 6. So 6*2 (each side) + 50 original dim

    zest()


def zest_ImageOps_gaussian_fitter():
    def it_fits_a_gaussian():
        xs, ys = np.mgrid[0:20, 0:20]
        data = imops._circ_gaussian(20.0, 10, 10, 1.0)(xs, ys) + 0.2 * np.random.random(
            xs.shape
        )
        center = YX(data.shape) / 2

        g_params = imops.fit_circ_gaussian(data)
        assert -1.0 < (20.0 - g_params[0]) < 1.0
        assert -1.0 < (10.0 - center.y - g_params[1]) < 1.0
        assert -1.0 < (10.0 - center.x - g_params[2]) < 1.0
        assert -0.15 < (1.0 - g_params[3]) < 0.15

    def it_exceptions_on_failures():
        data = np.zeros((20, 20))
        with zest.raises(Exception):
            imops.fit_circ_gaussian(data)

    zest()


def zest_region_enumerate():
    expected_coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [2.0, 3.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [3.0, 2.0],
            [3.0, 3.0],
        ]
    )

    def it_handles_single_image():
        im = np.zeros((512, 512))

        found_coords = []
        for win, y, x, coord in imops.region_enumerate(im, n_divs=4):
            assert win.shape == (128, 128)
            found_coords += [(coord // 128)]

        assert np.all(np.array(found_coords) == expected_coords)

    def it_handles_image_stack():
        im = np.zeros((5, 512, 512))

        found_coords = []
        for win, y, x, coord in imops.region_enumerate(im, n_divs=4):
            assert win.shape == (5, 128, 128)
            found_coords += [(coord // 128)]

        assert np.all(np.array(found_coords) == expected_coords)

    zest()


def zest_region_map():
    def it_handles_single_image():
        im = np.zeros((512, 512))

        def _func(im):
            if im.shape != (128, 128):
                raise Exception("Wrong shape")
            return 1.0

        res = imops.region_map(im, _func, divs=4)
        assert res.shape == (4, 4)
        assert np.all(res == 1.0)

    def it_handles_image_stack():
        im = np.zeros((10, 512, 512))

        def _func(im):
            if im.shape != (10, 128, 128):
                raise Exception("Wrong shape")
            return 1.0

        res = imops.region_map(im, _func, divs=4)
        assert res.shape == (4, 4)
        assert np.all(res == 1.0)

    zest()


def zest_rolling_window():
    def it_deals_with_even_divisibility():
        im = np.arange(6 * 6).reshape((6, 6))
        samples = imops.rolling_window(im, (2, 2), (3, 3))
        assert samples.shape == (3, 3, 2, 2)
        assert np.all(samples[0, 0] == [[0, 1], [6, 7]])
        assert np.all(samples[2, 2] == [[28, 29], [34, 35]])

    def it_deals_with_odd_divisibility():
        im = np.arange(5 * 6).reshape((5, 6))
        samples = imops.rolling_window(im, (3, 3), (2, 2))
        assert samples.shape == (2, 2, 3, 3)
        assert np.all(samples[0, 0] == [[0, 1, 2], [6, 7, 8], [12, 13, 14]])
        assert np.all(samples[1, 1] == [[15, 16, 17], [21, 22, 23], [27, 28, 29]])

    def it_raises_on_illegal_window():
        with zest.raises(ValueError):
            imops.rolling_window(np.zeros((10, 10)), (3, 3), (2, 2))

    def it_deals_with_a_stack():
        ims = np.stack(
            (
                1 * (np.arange(6 * 6) + 1).reshape((6, 6)),
                10 * (np.arange(6 * 6) + 1).reshape((6, 6)),
                100 * (np.arange(6 * 6) + 1).reshape((6, 6)),
            )
        )
        samples = imops.rolling_window(ims, (2, 2), (3, 3)).astype(int)
        assert samples.shape == (3, 3, 3, 2, 2)
        assert np.all(samples[0, 0, 0, :, :] == [[1, 2], [7, 8]])
        assert np.all(samples[2, 0, 0, :, :] == [[100, 200], [700, 800]])
        assert np.all(samples[0, 2, 2, :, :] == [[29, 30], [35, 36]])
        assert np.all(samples[2, 2, 2, :, :] == [[2900, 3000], [3500, 3600]])
        # debug(samples[2, 0, 0])
        # assert np.all(samples[2, 0, 0] == [[0, 1], [6, 7]])
        # assert np.all(samples[2, 2] == [[28, 29], [34, 35]])

    def it_returns_coords():
        im = np.arange(6 * 6).reshape((6, 6))
        _, coords = imops.rolling_window(im, (2, 2), (3, 3), return_coords=True)
        assert np.all(
            coords
            == [
                [[0, 0], [0, 2], [0, 4],],
                [[2, 0], [2, 2], [2, 4],],
                [[4, 0], [4, 2], [4, 4],],
            ]
        )

    zest()


def zest_interp():
    def _check(res):
        assert np.isclose(res[0, 0], 0.0)
        assert np.isclose(res[-1, 0], 2.0)
        assert np.isclose(res[0, -1], 2.0)
        assert np.isclose(res[-1, -1], 3.0)

    def it_interpolates_to_4_by_4_to_12_by_12():
        # Uses cubic interp
        out_size = 12
        src = np.array([[0, 1, 1, 2], [1, 1, 1, 2], [1, 1, 1, 2], [2, 2, 2, 3],])
        res = imops.interp(src, (out_size, out_size))
        assert res.shape == (12, 12)
        _check(res)

    def it_interpolates_to_3_by_3_to_4_by_4():
        # Requires a linear interp
        out_size = 4
        src = np.array([[0, 1, 2], [1, 1, 2], [2, 2, 3],])
        res = imops.interp(src, (out_size, out_size))
        assert res.shape == (4, 4)
        _check(res)

    def it_interpolates_to_2_by_2_to_4_by_4():
        # Requires a linear interp
        out_size = 4
        src = np.array([[0, 2], [2, 3],])
        res = imops.interp(src, (out_size, out_size))
        assert res.shape == (4, 4)
        _check(res)

    def it_interpolates_to_1_by_1_to_4_by_4():
        # Bypasses interp
        out_size = 4
        src = np.array([[3.0]])
        res = imops.interp(src, (out_size, out_size))
        assert res.shape == (4, 4)
        assert np.all(res == 3.0)

    zest()


def zest_gauss2_rho_form():
    def it_centers_odd():
        # When mea is odd and the center is at +0.5 the center
        # pixel should be the brightes
        true_params = (100.0, 1.5, 1.5, 5.5, 5.5, 0.0, 0.0, 11)
        im = imops.gauss2_rho_form(*true_params)
        assert np.argwhere(im == im.max()).tolist() == [[5, 5]]

    def it_centers_even():
        # When mea is even and the center is split between the inner four pixels
        # pixel should be the brightes
        true_params = (100.0, 1.5, 1.5, 5.0, 5.0, 0.0, 0.0, 10)
        im = imops.gauss2_rho_form(*true_params)
        assert np.argwhere(im == im.max()).tolist() == [[4, 4], [4, 5], [5, 4], [5, 5]]

    zest()


def zest_fit_gauss2():
    def it_fits_what_it_generates():
        true_params = (10.0, 1.5, 1.8, 8.5, 8.0, 0.2, 5.0, 17)
        im = imops.gauss2_rho_form(*true_params)
        fit_params, _ = imops.fit_gauss2(im)
        assert np.all((np.array(fit_params) - np.array(true_params)) ** 2 < 0.001 ** 2)

    zest()


def zest_distribution_aspect_ratio():
    def it_returns_1_for_circular():
        im = imops.gauss2_rho_form(10, 1.0, 1.0, 8, 8, rho=0.0, const=0, mea=17)
        ratio = imops.distribution_aspect_ratio(im)
        assert (ratio - 1.0) ** 2 < 0.001 ** 2

    def it_returns_3_for_rho_one_half():
        im = imops.gauss2_rho_form(10, 1.0, 1.0, 8, 8, rho=0.5, const=0, mea=17)
        ratio = imops.distribution_aspect_ratio(im)
        assert (ratio - 3.0) ** 2 < 0.1 ** 2

    def it_returns_2_for_x_ellipse():
        im = imops.gauss2_rho_form(10, 1.0, 1.5, 8, 8, rho=0.0, const=0, mea=17)
        ratio = imops.distribution_aspect_ratio(im)
        assert (ratio - 2.25) ** 2 < 0.01 ** 2

    zest()


def zest_sub_pixel_center():
    def it_centers():
        for x in np.linspace(-1, 1, 10):
            im = imops.gauss2_rho_form(
                10, 1.5, 1.5, 8.5 + x, 8.5, rho=0.0, const=0, mea=17
            )
            uncentered = imops.com(im)
            assert (uncentered[1] - 8.5 - x) ** 2 < 0.08 ** 2

            centered_im = imops.sub_pixel_center(im)
            centered = imops.com(centered_im)
            assert (centered[1] - 8.5) ** 2 < 0.08 ** 2

    zest()


def zest_sub_pixel_align():
    def _synth_cycles():
        n_peaks = 500
        amp = 6000
        dim = (256, 256)

        bg_mean = 0
        bg_std = 100
        n_cycles = 3
        with synth.Synth(overwrite=True, dim=dim, n_cycles=n_cycles) as s:
            peaks = (
                synth.PeaksModelGaussianCircular(n_peaks=n_peaks)
                .amps_constant(val=amp)
                .widths_uniform(width=1.8)
                .locs_randomize()
            )
            synth.CameraModel(bias=bg_mean, std=bg_std)
            s.zero_aln_offsets()
            cy_ims = s.render_chcy()[0, :]

        return cy_ims, peaks.locs

    def it_aligns():
        cy_ims, locs = _synth_cycles()

        # Simulate sub-pixel shift
        true_offsets = np.array([[0.0, 0.0], [1.235, 3.728], [-3.492, 8.911],])

        im0 = cy_ims[0]
        im1 = imops.sub_pixel_shift(cy_ims[1], true_offsets[1])
        im2 = imops.sub_pixel_shift(cy_ims[2], true_offsets[2])
        im_stack = np.stack((im0, im1, im2)).astype(np.float32)
        pred_offsets = imops.sub_pixel_align(im_stack, n_divs=2, precision=10)
        diff = pred_offsets - true_offsets
        assert np.all(np.abs(diff) <= 0.1)

    zest()
