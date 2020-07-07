import numpy as np
from zest import zest
from plumbum import local
from plaster.run.base_result import BaseResult, ArrayResult
from plaster.tools.utils import tmp
from plaster.tools.utils import utils
from plaster.tools.log.log import debug


class SimplePropertyResult(BaseResult):
    name = "simple_property"
    filename = "simple_property.pkl"

    required_props = dict(foo=int)


class ComplexPropertyResult(BaseResult):
    name = "complex_property"
    filename = "complex_property.pkl"

    required_props = dict(foo=int, arr=ArrayResult)


def zest_ArrayResult():
    def it_returns_an_open_array_without_overwrite():
        with tmp.tmp_folder(chdir=True):
            ar = ArrayResult("test1", shape=(10, 5), dtype=np.uint8, mode="w+")
            fp = ar.arr()
            ar[:] = np.arange(10 * 5).astype(np.uint8).reshape((10, 5))
            _fp = ar.arr()
            assert _fp is fp
            ar.flush()
            assert local.path("test1").stat().st_size == 10 * 5

    def it_resizes():
        with tmp.tmp_folder(chdir=True):
            ar = ArrayResult("test1", shape=(10, 5), dtype=np.uint8, mode="w+")
            ar[:] = np.arange(10 * 5).astype(np.uint8).reshape((10, 5))
            ar.reshape((4, 5))
            assert ar.shape == (4, 5)
            assert np.all(ar.arr() == np.arange(4 * 5).astype(np.uint8).reshape((4, 5)))

    zest()


def zest_base_result():
    def it_saves_and_loads_a_property_list():
        with tmp.tmp_folder(chdir=True):
            res1 = SimplePropertyResult(foo=2)
            res1.save()
            assert local.path(SimplePropertyResult.filename).exists()

            res2 = SimplePropertyResult.load_from_folder(".")
            assert res2.foo == 2

    def it_saves_and_loads_array_results():
        with tmp.tmp_folder() as folder:
            with local.cwd(folder):
                shape = (100, 87)
                arr = ArrayResult("arr.arr", dtype=np.float64, shape=shape, mode="w+")
                r = np.random.uniform(size=shape)
                arr[:] = r

                res1 = ComplexPropertyResult(foo=3, arr=arr)
                res1.save()

                pickle_file = local.path(ComplexPropertyResult.filename)
                assert (
                    pickle_file.stat().st_size < 200
                )  # The important part is that it doesn't include the array!

                arr_file = local.path("arr.arr")
                assert (
                    arr_file.stat().st_size == shape[0] * shape[1] * 8
                )  # 8 bytes for a float64

            # It should go back to a different folder
            # but the load_from_folder() should be able
            # deal with that
            assert local.cwd != folder

            res2 = ComplexPropertyResult.load_from_folder(folder)
            assert res2.foo == 3
            assert np.all(res2.arr == r)

    # def it_defaults_row_order():
    #     """
    #     I often do operations on the first dimension of matrix (rows)
    #     and I want to be sure that these are memory contiguous
    #     """
    #
    #     with tmp.tmp_folder(chdir=True):
    #         shape = (100_000_000, 2)  # 100_000_000 * 5 * 8 = 4.0 GB array!
    #         arr = ArrayResult("arr.arr", dtype=np.float64, shape=shape, mode="w+", order="F")
    #
    #         with utils.Timer("horizontal", show_start=True):
    #             n_blocks = 100_000
    #             block_size = shape[0] // n_blocks
    #             for block_i in range(n_blocks):
    #                 arr[block_i*block_size:(block_i+1)*block_size, :] = 1
    #             arr.flush()
    #
    #         with utils.Timer("vertical", show_start=True):
    #             arr[:, 0] = 1
    #             arr[:, 1] = 2
    #             arr.flush()

    zest()
