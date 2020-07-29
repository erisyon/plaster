from munch import Munch
import pandas as pd
import numpy as np
from plumbum import local
from plaster.run.base_result import BaseResult, ArrayResult
from plaster.run.ims_import.ims_import_params import ImsImportParams
from plaster.tools.utils import utils
from plaster.tools.utils.fancy_indexer import FancyIndexer
from plaster.tools.log.log import debug
from plaster.tools.schema import check


class ImsImportResult(BaseResult):
    """
    The ND2 importer results in a large number of image files, one per field.

    The normal result manifest contains a list of those filenames
    and a helper function lazy loads them.

    Follows the "Composite DataFrame Pattern"
    """

    name = "ims_import"
    filename = "ims_import.pkl"

    required_props = dict(
        params=ImsImportParams,
        n_fields=int,
        n_channels=int,
        n_cycles=int,
        dim=int,
        tsv_data=Munch,
        dtype=str,
    )

    _metadata_columns = (
        "x",
        "y",
        "z",
        "pfs_status",
        "pfs_offset",
        "exposure_time",
        "camera_temp",
        "field_i",
        "cycle_i",
    )

    def _field_ims_filename(self, field_i):
        return str(self._folder / f"field_{field_i:03d}.npy")

    def _field_metadata_filename(self, field_i):
        return str(self._folder / f"field_{field_i:03d}_metadata.pkl")

    def _field_qualities_filename(self, field_i):
        return str(self._folder / f"field_{field_i:03d}_qualities.pkl")

    def allocate_field(self, field_i, shape, dtype):
        filename = self._field_ims_filename(field_i)
        return ArrayResult(filename, dtype, shape, mode="w+")

    def save_field(
        self, field_i, field_chcy_ims, metadata_by_cycle=None, chcy_qualities=None
    ):
        """
        When using parallel field maps we can not save into the result
        because that will not be serialized back to the main thread.
        Rather, all field oriented results are written to a
        temporary pickle file and are reduced to a single value
        in the main thread's result instance.
        """
        check.t(field_chcy_ims, ArrayResult)
        field_chcy_ims.flush()

        if metadata_by_cycle is not None:
            utils.pickle_save(self._field_metadata_filename(field_i), metadata_by_cycle)

        if chcy_qualities is not None:
            utils.pickle_save(self._field_qualities_filename(field_i), chcy_qualities)

    def save(self):
        """
        Gather metadata that was written into temp files
        into one metadata_df and remove those files.
        """
        files = sorted(self._folder // "field_*_metadata.pkl")
        rows = [
            cycle_metadata
            for file in files
            for cycle_metadata in utils.pickle_load(file)
        ]

        for file in files:
            file.delete()

        self._metadata_df = pd.DataFrame(rows).rename(
            columns=dict(x="stage_x", y="stage_y", z="stage_z")
        )

        files = sorted(self._folder // "field_*_qualities.pkl")
        rows = []
        for field_i, file in enumerate(files):
            chcy_qualities = utils.pickle_load(file)
            for ch in range(self.n_channels):
                for cy in range(self.n_cycles):
                    rows += [(field_i, ch, cy, chcy_qualities[ch, cy])]

        for file in files:
            file.delete()

        self._qualities_df = pd.DataFrame(
            rows, columns=("field_i", "channel_i", "cycle_i", "quality")
        )

        super().save()

    def field_chcy_ims(self, field_i):
        if field_i not in self._cache_field_chcy_ims:
            self._cache_field_chcy_ims[field_i] = ArrayResult(
                self._field_ims_filename(field_i),
                dtype=np.dtype(self.dtype),
                shape=(self.n_channels, self.n_cycles, self.dim, self.dim),
            )
        return self._cache_field_chcy_ims[field_i].arr()

    def n_fields_channel_cycles(self):
        return self.n_fields, self.n_channels, self.n_cycles

    @property
    def ims(self):
        """Return a fancy-indexer that can return slices from [fields, channels, cycles]"""
        return FancyIndexer(
            (self.n_fields, self.n_channels, self.n_cycles),
            lookup_fn=lambda fl, ch, cy: self.field_chcy_ims(fl)[ch, cy],
        )

    def metadata(self):
        return self._metadata_df

    def qualities(self):
        return self._qualities_df

    def __init__(self, folder=None, is_loaded_result=False, **kwargs):
        super().__init__(folder, is_loaded_result=is_loaded_result, **kwargs)
        self._cache_field_chcy_ims = {}

    def __repr__(self):
        try:
            return f"ND2ImportResult with {self.n_fields} fields."
        except:
            return "ND2ImportResult"
