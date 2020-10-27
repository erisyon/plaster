import tempfile
from contextlib import contextmanager

import numpy as np
from munch import Munch
from plaster.run.ims_import import ims_import_worker as worker
from plaster.run.ims_import.ims_import_params import ImsImportParams
from plaster.tools.log.log import debug
from plumbum import local
from zest import zest


class _MockND2(Munch):
    def get_field(self, field, channel):
        ims = np.zeros((self.dim[0], self.dim[1]), dtype=np.uint16)
        if self._fill_by == "channel":
            ims[:, :] = channel
        elif self._fill_by == "cycle":
            ims[:, :] = field
        return ims


@contextmanager
def MockND2(**kwargs):
    yield _MockND2(**kwargs)


def zest_ims_import():
    tmp_src = tempfile.NamedTemporaryFile()
    tmp_dst = tempfile.TemporaryDirectory()

    src_path = local.path(tmp_src.name)
    with local.cwd(local.path(tmp_dst.name)):

        m_scan_nd2_files = zest.stack_mock(worker._scan_nd2_files)
        m_scan_tif_files = zest.stack_mock(worker._scan_tif_files)
        m_scan_npy_files = zest.stack_mock(worker._scan_npy_files)
        m_nd2 = zest.stack_mock(worker._nd2)

        n_cycles = 2
        n_fields = 3
        n_channels = 4
        cycle_files = [(src_path / f"{i}.nd2") for i in range(n_cycles)]

        def _make_nd2(dim, fill_by="channel", n_cycles=None):
            return MockND2(
                n_fields=n_fields,
                n_channels=n_channels,
                dim=(dim, dim),
                x=[0] * n_fields,
                y=[0] * n_fields,
                z=[0] * n_fields,
                pfs_status=[0] * n_fields,
                pfs_offset=[0] * n_fields,
                exposure_time=[0] * n_fields,
                camera_temp=[0] * n_fields,
                _fill_by=fill_by,
                _n_cycles=n_cycles,
            )

        ims_import_params = None
        nd2 = None

        def _before():
            nonlocal ims_import_params, nd2
            ims_import_params = ImsImportParams()
            m_nd2.hook_to_call = lambda _: _make_nd2(64)
            m_scan_nd2_files.returns(cycle_files)
            m_scan_tif_files.returns([])
            m_scan_npy_files.returns([])

        def it_scatter_gathers():
            result = worker.ims_import(src_path, ims_import_params)
            emitted_files = list(local.path(".").walk())
            assert len(emitted_files) == 9
            assert result.params == ims_import_params
            assert result.n_fields == n_fields
            assert result.n_channels == n_channels
            assert result.n_cycles == n_cycles

        def it_converts_to_power_of_2():
            with zest.mock(worker._convert_message):
                m_nd2.hook_to_call = lambda _: _make_nd2(63)
                result = worker.ims_import(src_path, ims_import_params)
                assert result.field_chcy_ims(0).shape == (n_channels, n_cycles, 64, 64)

        def it_limits_fields():
            ims_import_params.n_fields_limit = 1
            result = worker.ims_import(src_path, ims_import_params)
            assert result.n_fields == 1

        def it_imports_src_channels():
            result = worker.ims_import(src_path, ims_import_params)
            assert np.all(result.field_chcy_ims(0)[0, :, :, :] == 0.0)
            assert np.all(result.field_chcy_ims(0)[1, :, :, :] == 1.0)

        def it_can_skip_fields():
            ims_import_params.start_field = 1
            result = worker.ims_import(src_path, ims_import_params)
            assert result.n_fields == 2

        def it_can_limit_cycles():
            ims_import_params.n_cycles_limit = 1
            result = worker.ims_import(src_path, ims_import_params)
            assert result.n_fields == n_fields
            assert result.n_cycles == n_cycles - 1
            assert result.n_channels == n_channels

        def it_can_skip_cycles():
            ims_import_params.start_cycle = 1
            result = worker.ims_import(src_path, ims_import_params)
            assert result.n_fields == n_fields
            assert result.n_cycles == n_cycles - 1
            assert result.n_channels == n_channels

        def it_respects_channel_map():
            ims_import_params.dst_ch_i_to_src_ch_i = [1, 0]
            result = worker.ims_import(src_path, ims_import_params)
            assert result.n_fields == n_fields
            assert result.n_cycles == n_cycles
            assert result.n_channels == 2
            assert np.all(result.field_chcy_ims(0)[0, :, :, :] == float(1))
            assert np.all(result.field_chcy_ims(0)[1, :, :, :] == float(0))

        def movies():
            def _before():
                nonlocal ims_import_params, nd2
                ims_import_params = ImsImportParams(is_movie=True)
                m_nd2.hook_to_call = lambda _: _make_nd2(64, "cycle", n_fields)
                m_scan_nd2_files.returns(cycle_files)
                m_scan_tif_files.returns([])

            def it_swaps_fields_cycles():
                result = worker.ims_import(src_path, ims_import_params)
                assert result.n_cycles == n_fields
                assert result.n_fields == n_cycles
                assert result.n_channels == n_channels
                for cy in range(result.n_cycles):
                    assert np.all(result.field_chcy_ims(0)[:, cy, :, :] == float(cy))

            def it_can_limit_cycles():
                ims_import_params.n_cycles_limit = 2
                result = worker.ims_import(src_path, ims_import_params)
                assert result.n_cycles == 2
                assert result.n_fields == n_cycles
                assert result.n_channels == n_channels
                for cy in range(result.n_cycles):
                    assert np.all(result.field_chcy_ims(0)[:, cy, :, :] == float(cy))

            def it_can_skip_cycles():
                ims_import_params.start_cycle = 1
                result = worker.ims_import(src_path, ims_import_params)
                assert result.n_cycles == n_fields - 1
                assert result.n_fields == n_cycles
                assert result.n_channels == n_channels
                for cy in range(result.n_cycles):
                    assert np.all(
                        result.field_chcy_ims(0)[:, cy, :, :] == float(cy + 1)
                    )

            def it_converts_to_power_of_2():
                with zest.mock(worker._convert_message):
                    m_nd2.hook_to_call = lambda _: _make_nd2(63, "cycle", n_fields)
                    result = worker.ims_import(src_path, ims_import_params)
                    assert result.field_chcy_ims(0).shape == (
                        n_channels,
                        n_fields,
                        64,
                        64,
                    )

            def it_respects_channel_map():
                nonlocal ims_import_params, nd2
                ims_import_params = ImsImportParams(is_movie=True)
                m_nd2.hook_to_call = lambda _: _make_nd2(64)  # Channel mode
                m_scan_nd2_files.returns(cycle_files)
                m_scan_tif_files.returns([])

                ims_import_params.dst_ch_i_to_src_ch_i = [1, 0]
                result = worker.ims_import(src_path, ims_import_params)
                assert result.n_cycles == n_fields
                assert result.n_fields == n_cycles
                assert result.n_channels == 2
                assert np.all(result.field_chcy_ims(0)[0, :, :, :] == float(1))
                assert np.all(result.field_chcy_ims(0)[1, :, :, :] == float(0))

            zest()

        zest()


def zest_ims_import_from_npy():
    tmp_dst = tempfile.TemporaryDirectory()
    with local.cwd(local.path(tmp_dst.name)):
        m_scan_nd2_files = zest.stack_mock(worker._scan_nd2_files)
        m_scan_tif_files = zest.stack_mock(worker._scan_tif_files)
        m_scan_npy_files = zest.stack_mock(worker._scan_npy_files)
        m_load_npy = zest.stack_mock(worker._load_npy)

        npy_files = [
            # area, field, channel, cycle
            "area_000_cell_000_555nm_001.npy",
            "area_000_cell_000_647nm_001.npy",
            "area_000_cell_000_555nm_002.npy",
            "area_000_cell_000_647nm_002.npy",
            "area_000_cell_000_555nm_003.npy",
            "area_000_cell_000_647nm_003.npy",
            "area_000_cell_001_555nm_001.npy",
            "area_000_cell_001_647nm_001.npy",
            "area_000_cell_001_555nm_002.npy",
            "area_000_cell_001_647nm_002.npy",
            "area_000_cell_001_555nm_003.npy",
            "area_000_cell_001_647nm_003.npy",
        ]

        ims_import_params = None

        def _before():
            nonlocal ims_import_params
            ims_import_params = ImsImportParams()
            m_scan_nd2_files.returns([])
            m_scan_tif_files.returns([])
            m_scan_npy_files.returns(npy_files)
            m_load_npy.returns(np.zeros((16, 16)))

        def it_scans_npy_arrays():
            scan_result = worker._scan_files("")

            assert scan_result.mode == worker.ScanFileMode.npy
            assert scan_result.nd2_paths == []
            assert scan_result.tif_paths_by_field_channel_cycle == {}
            assert (
                local.path(scan_result.npy_paths_by_field_channel_cycle[(0, 0, 0)]).name
                == npy_files[0]
            )
            assert (
                scan_result.n_fields == 2
                and scan_result.n_channels == 2
                and scan_result.n_cycles == 3
            )
            assert scan_result.dim == (16, 16)

        def it_ims_import_npy():
            res = worker.ims_import(
                ".", ims_import_params, progress=None, pipeline=None
            )
            assert res.n_fields == 2 and res.n_channels == 2 and res.n_cycles == 3

        zest()
