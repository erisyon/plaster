"""
Import image files

Options:

    * Nikon ND2 files.

        Each ND2 file is a collection of channels/field images per cycle.
        This organization needs to be transposed to effectively parallelize the
        sigprocv2 stage (which acts on all cycles/channels of 1 field in parallel).

        Done in two stages:
            1. scatter the .nd2 files into individual .npy files
            2. gather those .nd2 files back into field stacks.

    * TIF files
"""

from skimage.io import imread
import os
import re
import numpy as np
from plaster.run.nd2 import ND2
from munch import Munch
from plumbum import local
from plaster.run.ims_import.ims_import_result import ImsImportResult
from plaster.tools.utils import utils
from plaster.tools.schema import check
from plaster.tools.zap import zap
from plaster.tools.tsv import tsv
from plaster.tools.log.log import important, debug, info
from plaster.tools.image import imops


def _scan_nd2_files(src_dir):
    """Mock-point"""
    return list(src_dir // "*.nd2")


def _scan_tif_files(src_dir):
    """Mock-point"""
    return list(src_dir.walk(filter=lambda f: f.suffix == ".tif"))


def _scan_npy_files(src_dir):
    """Mock-point"""
    return list(src_dir.walk(filter=lambda f: f.suffix == ".npy"))


def _nd2(src_path):
    """Mock-point"""
    return ND2(src_path)


def _load_npy(npy_path):
    """Mock-point"""
    return np.load(str(npy_path))


def _convert_message(target_dim, new_dim):
    """Mock-point"""
    important(f"Converting from dim {target_dim} to {new_dim}")


def _scan_files(src_dir):
    """
    Search for .nd2 (non-recursive) or .tif files (recursively) or .npy (non-recursive)

    For .npy the he naming convention is:
        area, field, channel, cycle
        examples:
        area_000_cell_000_555nm_001.npy
        area_000_cell_000_647nm_001.npy
    """
    nd2_paths = sorted(_scan_nd2_files(src_dir))
    tif_paths = sorted(_scan_tif_files(src_dir))
    npy_paths = sorted(_scan_npy_files(src_dir))

    tif_paths_by_field_channel_cycle = {}
    npy_paths_by_field_channel_cycle = {}
    n_fields = 0
    n_channels = 0
    n_cycles = 0
    min_field = 10000
    min_channel = 10000
    min_cycle = 10000
    dim = 0
    mode = None

    if len(nd2_paths) > 0:
        mode = "nd2"

        # OPEN a single image to get the vitals
        nd2 = _nd2(nd2_paths[0])
        n_fields = nd2.n_fields
        n_channels = nd2.n_channels
        dim = nd2.dim

    elif len(npy_paths) > 0:
        mode = "npy"

        area_cells = set()
        channels = set()
        cycles = set()

        # area_000_cell_000_555nm_001.npy
        npy_pat = re.compile(
            r"area_(?P<area>\d+)_cell_(?P<cell>\d+)_(?P<channel>\d+)nm_(?P<cycle>\d+)\.npy"
        )

        # PARSE the path names to determine channel, field, cycle
        for p in npy_paths:
            m = npy_pat.search(str(p))
            if m:
                found = Munch(m.groupdict())
                area_cells.add((int(found.area), int(found.cell)))
                channels.add(int(found.channel))
                cycles.add(int(found.cycle))
            else:
                raise ValueError(
                    f"npy file found ('{str(p)}') that did not match expected pattern."
                )

        cycle_by_cycle_i = {
            cycle_i: cycle_name for cycle_i, cycle_name in enumerate(sorted(cycles))
        }
        n_cycles = len(cycle_by_cycle_i)

        channel_by_channel_i = {
            channel_i: channel_name
            for channel_i, channel_name in enumerate(sorted(channels))
        }
        n_channels = len(channel_by_channel_i)

        area_cell_by_field_i = {
            field_i: area_cell for field_i, area_cell in enumerate(sorted(area_cells))
        }
        n_fields = len(area_cell_by_field_i)

        for field_i in range(n_fields):
            area, cell = area_cell_by_field_i[field_i]
            for channel_i in range(n_channels):
                channel = channel_by_channel_i[channel_i]
                for cycle_i in range(n_cycles):
                    cycle = cycle_by_cycle_i[cycle_i]
                    npy_paths_by_field_channel_cycle[(field_i, channel_i, cycle_i)] = (
                        local.path(src_dir)
                        / f"area_{area:03d}_cell_{cell:03d}_{channel}nm_{cycle:03d}.npy"
                    )

        # OPEN a single image to get the vitals
        im = _load_npy(str(npy_paths[0]))
        assert im.ndim == 2
        dim = im.shape

    elif len(tif_paths) > 0:
        mode = "tif"

        tif_pat = re.compile(
            r"_c(\d+)/img_channel(\d+)_position(\d+)_time\d+_z\d+\.tif"
        )

        # PARSE the path names to determine channel, field,
        for p in tif_paths:
            m = tif_pat.search(str(p))
            if m:
                cycle_i = int(m[1])
                channel_i = int(m[2])
                field_i = int(m[3])
                n_channels = max(channel_i, n_channels)
                n_cycles = max(cycle_i, n_cycles)
                n_fields = max(field_i, n_fields)
                min_field = min(field_i, min_field)
                min_channel = min(channel_i, min_channel)
                min_cycle = min(cycle_i, min_cycle)
                tif_paths_by_field_channel_cycle[(field_i, channel_i, cycle_i)] = p
            else:
                raise ValueError(
                    f"tif file found ('{str(p)}') that did not match expected pattern."
                )

        assert min_channel == 0
        n_channels += 1

        assert min_field == 0
        n_fields += 1

        if min_cycle == 0:
            n_cycles += 1
        elif min_cycle == 1:
            _tifs = {}
            for field_i in range(n_fields):
                for channel_i in range(n_channels):
                    for target_cycle_i in range(n_cycles):
                        _tifs[
                            (field_i, channel_i, target_cycle_i)
                        ] = tif_paths_by_field_channel_cycle[
                            (field_i, channel_i, target_cycle_i + 1)
                        ]
            tif_paths_by_field_channel_cycle = _tifs
        else:
            raise ValueError("tif cycle needs to start at 0 or 1")

        # OPEN a single image to get the vitals
        im = imread(str(tif_paths[0]))
        dim = im.shape
    else:
        raise ValueError(f"No image files (.nd2, .tif) were found in '{src_dir}'")

    return (
        mode,
        nd2_paths,
        tif_paths_by_field_channel_cycle,
        npy_paths_by_field_channel_cycle,
        n_fields,
        n_channels,
        n_cycles,
        dim,
    )


def _npy_filename_by_field_channel_cycle(field, channel, cycle):
    return f"__{field:03d}-{channel:02d}-{cycle:02d}.npy"


def _metadata_filename_by_field_cycle(field, cycle):
    return f"__{field:03d}-{cycle:02d}.json"


def _do_nd2_scatter(src_path, start_field, n_fields, cycle_i, n_channels, target_dim):
    """
    Scatter a cycle .nd2 into individual numpy files.

    target_dim is a scalar. The target will be put into this square form.
    """
    nd2 = _nd2(src_path)

    ims = nd2.get_fields()
    _n_channels = ims.shape[1]
    actual_dim = ims.shape[2:4]
    assert n_channels == _n_channels

    check.affirm(
        actual_dim[0] <= target_dim and actual_dim[1] <= target_dim,
        f"nd2 scatter requested {target_dim} which is smaller than {actual_dim}",
    )

    if actual_dim[0] != target_dim or actual_dim[1] != target_dim:
        # CONVERT into a zero pad
        new_ims = np.zeros(
            (n_fields, _n_channels, target_dim, target_dim), dtype=ims.dtype
        )
        new_ims[:, :, 0 : actual_dim[0], 0 : actual_dim[1]] = ims[:, :, :, :]
        ims = new_ims

    dst_files = []
    for field_i in range(start_field, start_field + n_fields):
        info = Munch(
            x=nd2.x[field_i],
            y=nd2.y[field_i],
            z=nd2.z[field_i],
            pfs_status=nd2.pfs_status[field_i],
            pfs_offset=nd2.pfs_offset[field_i],
            exposure_time=nd2.exposure_time[field_i],
            camera_temp=nd2.camera_temp[field_i],
            cycle_i=cycle_i,
            field_i=field_i,
        )
        info_dst_file = _metadata_filename_by_field_cycle(field_i, cycle_i)
        utils.json_save(info_dst_file, info)

        for channel_i in range(n_channels):
            dst_file = _npy_filename_by_field_channel_cycle(field_i, channel_i, cycle_i)
            dst_files += [dst_file]
            np.save(dst_file, ims[field_i, channel_i])
    return dst_files


def _do_tif_scatter(field_i, channel_i, cycle_i, path):
    im = imread(str(path))
    dst_file = _npy_filename_by_field_channel_cycle(field_i, channel_i, cycle_i)
    np.save(dst_file, im)
    return dst_file


def _quality(im):
    """
    Quality of an image by spatial low-pass filter.
    High quality images are one where there is very little
    low-frequency (but above DC) bands.
    """
    return imops.low_frequency_power(im, dim_half=3)


def _do_gather(
    input_field_i,
    output_field_i,
    src_channels,
    start_cycle,
    n_cycles,
    dim,
    nd2_import_result,
    mode,
    npy_paths_by_field_channel_cycle,
):
    """Gather a field"""
    check.list_t(src_channels, int)
    n_channels = len(src_channels)

    field_chcy_ims = np.zeros((n_channels, n_cycles, dim, dim))
    chcy_i_to_quality = np.zeros((n_channels, n_cycles))
    cy_i_to_metadata = [None] * n_cycles

    output_cycle_i = 0
    for input_cycle_i in range(start_cycle, n_cycles):
        # GATHER channels
        for dst_channel_i, src_channel_i in enumerate(src_channels):
            if mode == "npy":
                # These are being imported by npy originally with a different naming
                # convention than the scattered files.
                scatter_fp = npy_paths_by_field_channel_cycle[
                    (input_field_i, src_channel_i, input_cycle_i)
                ]
            else:
                scatter_fp = _npy_filename_by_field_channel_cycle(
                    input_field_i, src_channel_i, input_cycle_i
                )
            im = field_chcy_ims[dst_channel_i, output_cycle_i, :, :] = _load_npy(
                scatter_fp
            )
            chcy_i_to_quality[dst_channel_i, output_cycle_i] = _quality(im)

        # GATHER metadata files if any
        cy_i_to_metadata[output_cycle_i] = None
        try:
            cy_i_to_metadata[output_cycle_i] = utils.json_load_munch(
                _metadata_filename_by_field_cycle(input_field_i, input_cycle_i)
            )
        except FileNotFoundError:
            pass

        output_cycle_i += 1

    nd2_import_result.save_field(
        output_field_i, field_chcy_ims, cy_i_to_metadata, chcy_i_to_quality
    )

    return output_field_i


def _do_movie_import(
    nd2_path, output_field_i, start_cycle, n_cycles, target_dim, nd2_import_result
):
    """
    Import Nikon ND2 "movie" files.

    In this mode, each .nd2 file is a collection of images taken sequentially for a single field.
    This is in contrast to the typical mode where each .nd2 file is a chemical cycle spanning
    all fields/channels.

    Since all data for a given field is already in a single file, the parallel
    scatter/gather employed by the "normal" ND2 import task is not necessary.

    The "fields" from the .nd2 file become "cycles" as if the instrument had
    taken 1 field with a lot of cycles.
    """

    nd2 = _nd2(nd2_path)

    ims = nd2.get_fields()
    n_actual_cycles = ims.shape[0]
    n_channels = ims.shape[1]
    actual_dim = ims.shape[2:4]

    # The .nd2 file is usually of shape (n_fields, n_channels, dim, dim)
    # but in a movie, the n_fields is becoming the n_cycles so swap the fields and channel
    # putting ims into (n_channels, n_cycles, dim, dim)
    chcy_ims = np.swapaxes(ims, 0, 1)

    assert start_cycle + n_cycles <= n_actual_cycles
    chcy_ims = chcy_ims[:, start_cycle : start_cycle + n_cycles, :, :]

    check.affirm(
        actual_dim[0] <= target_dim and actual_dim[1] <= target_dim,
        f"nd2 scatter requested {target_dim} which is smaller than {actual_dim}",
    )

    if actual_dim[0] != target_dim or actual_dim[1] != target_dim:
        # CONVERT into a zero pad
        new_chcy_ims = np.zeros(
            (n_channels, n_cycles, target_dim, target_dim), dtype=ims.dtype
        )
        new_chcy_ims[:, :, 0 : actual_dim[0], 0 : actual_dim[1]] = chcy_ims[:, :, :, :]
        chcy_ims = new_chcy_ims

    # TODO Add quality

    nd2_import_result.save_field(output_field_i, chcy_ims)

    return output_field_i, n_actual_cycles


def ims_import(src_dir, ims_import_params, progress=None, pipeline=None):
    (
        mode,
        nd2_paths,
        tif_paths_by_field_channel_cycle,
        npy_paths_by_field_channel_cycle,
        n_fields_true,
        n_channels,
        n_cycles_true,
        dim,
    ) = _scan_files(src_dir)

    target_dim = max(dim[0], dim[1])

    if not utils.is_power_of_2(target_dim):
        new_dim = utils.next_power_of_2(target_dim)
        _convert_message(target_dim, new_dim)
        target_dim = new_dim

    src_channels = list(range(n_channels))

    def clamp_fields(n_fields_true):
        n_fields = n_fields_true
        n_fields_limit = ims_import_params.get("n_fields_limit")
        if n_fields_limit is not None:
            n_fields = n_fields_limit

        start_field = ims_import_params.get("start_field", 0)
        if start_field + n_fields > n_fields_true:
            n_fields = n_fields_true - start_field

        return start_field, n_fields

    def clamp_cycles(n_cycles_true):
        n_cycles = n_cycles_true
        n_cycles_limit = ims_import_params.get("n_cycles_limit")
        if n_cycles_limit is not None:
            n_cycles = n_cycles_limit

        start_cycle = ims_import_params.get("start_cycle", 0)
        if start_cycle + n_cycles > n_cycles_true:
            n_cycles = n_cycles_true - start_cycle

        return start_cycle, n_cycles

    tsv_data = tsv.load_tsv_for_folder(src_dir)
    ims_import_result = ImsImportResult(
        params=ims_import_params, tsv_data=Munch(tsv_data)
    )

    if ims_import_params.is_movie:
        start_field, n_fields = clamp_fields(len(nd2_paths))

        # In movie mode, the n_fields from the .nd2 file is becoming n_cycles
        n_cycles_true = n_fields_true
        start_cycle, n_cycles = clamp_cycles(n_cycles_true)

        field_iz, n_cycles_found = zap.arrays(
            _do_movie_import,
            dict(
                nd2_path=nd2_paths[start_field : start_field + n_fields],
                output_field_i=list(range(n_fields)),
            ),
            _process_mode=True,
            _progress=progress,
            _stack=True,
            start_cycle=start_cycle,
            n_cycles=n_cycles,
            target_dim=target_dim,
            nd2_import_result=ims_import_result,
        )

    else:
        start_field, n_fields = clamp_fields(n_fields_true)

        if pipeline:
            pipeline.set_phase(0, 2)

        if mode == "nd2":
            n_cycles_true = len(nd2_paths)

            # SCATTER
            zap.arrays(
                _do_nd2_scatter,
                dict(cycle_i=list(range(len(nd2_paths))), src_path=nd2_paths),
                _process_mode=True,
                _progress=progress,
                _stack=True,
                start_field=start_field,
                n_fields=n_fields,
                n_channels=n_channels,
                target_dim=target_dim,
            )

        elif mode == "tif":
            # SCATTER
            work_orders = [
                Munch(field_i=k[0], channel_i=k[1], cycle_i=k[2], path=path)
                for k, path in tif_paths_by_field_channel_cycle.items()
            ]
            results = zap.work_orders(
                _do_tif_scatter, work_orders, _trap_exceptions=False
            )

            # CHECK that every file exists
            for f in range(n_fields):
                for ch in range(n_channels):
                    for cy in range(n_cycles_true):
                        expected = f"__{f:03d}-{ch:02d}-{cy:02d}.npy"
                        if expected not in results:
                            raise FileNotFoundError(
                                f"File is missing in tif pattern: {expected}"
                            )

        elif mode == "npy":
            # In npy mode there's no scatter as the files are already fully scattered
            pass

        else:
            raise ValueError(f"Unknown im import mode {mode}")

        if pipeline:
            pipeline.set_phase(1, 2)

        # GATHER
        start_cycle, n_cycles = clamp_cycles(n_cycles_true)

        field_iz = zap.arrays(
            _do_gather,
            dict(
                input_field_i=list(range(start_field, start_field + n_fields)),
                output_field_i=list(range(0, n_fields)),
            ),
            _process_mode=True,
            _progress=progress,
            _stack=True,
            src_channels=src_channels,
            start_cycle=start_cycle,
            n_cycles=n_cycles,
            dim=target_dim,
            nd2_import_result=ims_import_result,
            mode=mode,
            npy_paths_by_field_channel_cycle=npy_paths_by_field_channel_cycle,
        )

    ims_import_result.n_fields = len(field_iz)
    ims_import_result.n_channels = n_channels
    ims_import_result.n_cycles = n_cycles
    ims_import_result.dim = target_dim

    # CLEAN
    for file in local.cwd // "__*":
        file.delete()

    return ims_import_result
