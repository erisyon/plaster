"""
Import image files

Options:

    * Nikon ND2 files.

        Each ND2 file is a collection of channels/field images per cycle.
        This organization needs to be transposed to effectively parallelize the
        sigproc stage (which acts on all cycles/channels of 1 field in parallel).

        Done in two stages:
            1. scatter the .nd2 files into individual .npy files
            2. gather those .nd2 files back into field stacks.

    * TIF files


Work needed Jul 2020

This module is responsible for accepting image input from
a variety of input formats and converting it into numpy
arrays in the shape and organization that are convenient for
downstream processing.

Definitions:
    * Frame / image
        An image taken by the microscope. This is at one wavelength,
        at one (X,Y,Z) position of the microscope's stage. It is
        typcially a square power-of-two and 16-bits but may deviate
        from this and need correction to the next square power-of-two
        and may need to be padded up to 16-bits.
    * Channel
        A set of frames that correspond to a certain wavelength (ie
        filter setting) on the scope.
    * Cycle
        A set of frames (comprising all channels, all fields) that
        are taken after a certain "chemical cycle"
    * Field
        A field is all frames (all channels, cycles) that coorespond
        to a given (x, y, z) position of the microscope's stage.
    * Metadata
        Various metadata about the camera. Example: focus, brightess, etc.
        Not consistent on all input formats and scopes
    * "mea" or "measure" is a 1-dimensional measure
    * "dim" is a 2-d measure. If something is square then dim == (mea, mea)

Input formats:
    *.nd2
        This is a Nikon format the some of our older scopes use.
        It is not a well supported nor documented format but
        after some significant reverse-engineering I was able to
        get most of what I wanted out of the files. See nd2.py.

        ND2 files are usually, BUT NOT ALWAYS, written in the
        order that 1 nd2 file contains 1 cycle (all channels).

        The later steps of processing want the data in 1-file-per-field
        order so that each field can be processed in parallel.

    *.tif
        This is an even older use-case where some of the earliest
        experiments dumped large numbers of 16-but tif files with magic
        semantically significant filenames.  At least TIF is relatively
        well-supported and can use the skimage library.
        The tif files are sometimes spread out over a directory
        tree and require recursive traversal to find them all.

    *.npy
        This is the simplest format and will be what our production
        scopes will emit (to hopefully avoid a lot of the work
        in this module!)

    Input considerations:
        * The order of the input data / files is not typically in
          the most convenient order for processing.
        * The order of the input data is not always consistent and
          various modes have to accommodate this.
        * The input frames are not always in a power-of-two and
          have to be converted
        * A quality metric is useful to have and we might as well
          calculate it while we have the frames in memory

Output format:
    The output is in .npy format, all frames correctly scale to a power of two
    and organized by field.


Current approach:
    If you are in "movie mode" (which is an unhelpful name and needs to be
    revisited) then the .nd2 files are already in field-major order and
    therefore the task is simpler.

    If NOT in movie mode then a scatter/gather approach is taken:
        1. Scan file system and use some hints provided by the ims_import_params
           to decide which file-names/file-paths will be imported
        2. Scatter files by deconstructing ND2 or TIF files into individual
           frames (1 file per frame) and converting them to next power-of-two
           if needed.
        3. Gather files by copying individual scattered files into the
           correct output order (field, channel, cycle).

Other notes
    * TODO Explain clamping and note that the input start/stop is not nec same as output
    * TODO Explain zap
    * As currently implemented even the scanning of the files is a bit
      slow as it opens and checks vitals on every file when it could
      do that progressively.
    * It might be faster to avoid the scatter stage and instead have
      each gather thread/process open the source files and scan them
      to find the desired frame.
    * The dimension conversions are painful
    * I use memory mapping to keep memory requiremetns down
    * Things are generally hard-coded to expect 16-bit files in places
      and that's okay as we do not expect other formats but it
      would be nice to be cleaner about it.
    * Some of the metadata is accompaniyed by TSV (tab separated value) files
      of an unusual format.


"""

from skimage.io import imread
import re
import numpy as np
from plaster.run.ims_import.nd2 import ND2
from munch import Munch
from plumbum import local
from plaster.run.ims_import.ims_import_result import ImsImportResult
from plaster.tools.utils import utils
from plaster.tools.schema import check
from plaster.tools.zap import zap
from plaster.tools.tsv import tsv
from plaster.tools.log.log import important, debug, info, prof
from plaster.tools.image import imops


OUTPUT_NP_TYPE = np.float32

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


def _convert_message(target_mea, new_dim):
    """Mock-point"""
    # important(f"Converting from dim {target_mea} to {new_dim}")
    pass


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

    if len(nd2_paths) > 0:
        mode = "nd2"

        # OPEN a single image to get the vitals
        with _nd2(nd2_paths[0]) as nd2:
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


def _do_nd2_scatter(src_path, start_field, n_fields, cycle_i, n_channels, target_mea):
    """
    Scatter a cycle .nd2 into individual numpy files.

    target_mea is a scalar. The target will be put into this square form.
    """

    working_im = np.zeros((target_mea, target_mea), np.uint16)

    with _nd2(src_path) as nd2:
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
                im = nd2.get_field(field_i, channel_i)

                if im.shape[0] != target_mea or im.shape[1] != target_mea:
                    working_im[0: im.shape[0], 0: im.shape[1]] = im[:, :]
                    im = working_im

                dst_file = _npy_filename_by_field_channel_cycle(field_i, channel_i, cycle_i)
                dst_files += [dst_file]
                np.save(dst_file, im)

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

    field_chcy_arr = nd2_import_result.allocate_field(output_field_i, (n_channels, n_cycles, dim, dim), OUTPUT_NP_TYPE)
    field_chcy_ims = field_chcy_arr.arr()

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

            im = _load_npy(scatter_fp)
            if im.dtype != OUTPUT_NP_TYPE:
                im = im.astype(OUTPUT_NP_TYPE)
            field_chcy_ims[dst_channel_i, output_cycle_i, :, :] = im
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
        output_field_i, field_chcy_arr, cy_i_to_metadata, chcy_i_to_quality
    )

    return output_field_i


def _do_movie_import(
    nd2_path, output_field_i, start_cycle, n_cycles, target_mea, nd2_import_result
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

    with _nd2(nd2_path) as nd2:

        # TODO FIX ME
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
            actual_dim[0] <= target_mea and actual_dim[1] <= target_mea,
            f"nd2 scatter requested {target_mea} which is smaller than {actual_dim}",
        )

        if actual_dim[0] != target_mea or actual_dim[1] != target_mea:
            # CONVERT into a zero pad
            new_chcy_ims = np.zeros(
                (n_channels, n_cycles, target_mea, target_mea), dtype=ims.dtype
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

    target_mea = max(dim[0], dim[1])

    if not utils.is_power_of_2(target_mea):
        new_dim = utils.next_power_of_2(target_mea)
        _convert_message(target_mea, new_dim)
        target_mea = new_dim

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

    # ALLOCATE the ImsImportResult
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
            target_mea=target_mea,
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
                _process_mode=False,
                _progress=progress,
                _stack=True,
                start_field=start_field,
                n_fields=n_fields,
                n_channels=n_channels,
                target_mea=target_mea,
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
            dim=target_mea,
            nd2_import_result=ims_import_result,
            mode=mode,
            npy_paths_by_field_channel_cycle=npy_paths_by_field_channel_cycle,
        )

    ims_import_result.n_fields = len(field_iz)
    ims_import_result.n_channels = n_channels
    ims_import_result.n_cycles = n_cycles
    ims_import_result.dim = target_mea
    ims_import_result.dtype = np.dtype(OUTPUT_NP_TYPE).name

    # CLEAN
    for file in local.cwd // "__*":
        file.delete()

    return ims_import_result
