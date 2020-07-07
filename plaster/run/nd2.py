"""
A hand-crafted ND2 reader that is much faster and more stable than others
that I found online but doesn't support all the crazy modes that ND2 files
support. Only supports 16-bit forms.

Helpful reference:
https://github.com/openmicroscopy/bioformats/blob/develop/components/formats-gpl/src/loci/formats/in/NativeND2Reader.java
"""


import numpy as np
import struct
from munch import Munch


class ND2:
    data_types = {
        1: "unsigned_char",
        2: "unsigned_int",
        3: "unsigned_int_2",
        5: "unsigned_long",
        6: "double",
        8: "string",
        9: "char_array",
        11: "dict",
    }

    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()

        sig = bytes("ND2 CHUNK MAP SIGNATURE 0000001!", encoding="utf-8")
        off_sig = self.data.find(sig)

        chunk_map_pos = self._u64(off_sig + len(sig))

        tmp_len_two = self._u32(chunk_map_pos + 4)
        chunk_map_length = self._u64(chunk_map_pos + 8)
        chunk_map_pos += 16 + tmp_len_two
        chunk_map_end = chunk_map_pos + chunk_map_length

        name_terminator = bytes("!", encoding="utf-8")
        off = chunk_map_pos

        blocks = []
        self.images = {}
        while off < chunk_map_end:
            term = self.data[off:].find(name_terminator)
            name = self.data[off : off + term + 1]
            off += term + 1
            pos = self._u64(off)
            off += 8
            cnt = self._u64(off)
            off += 8
            if name == sig:
                break

            name = str(name[:-1], encoding="utf-8")
            block = Munch(pos=pos, cnt=cnt, name=name)
            blocks += [block]
            if name.startswith("ImageDataSeq|"):
                field = int(name[13:])
                self.images[field] = block

        for block in blocks:
            if block.name.startswith("ImageAttributesLV"):
                d = self._metadata_section(block)
                self.n_channels = d.SLxImageAttributes.uiVirtualComponents
                assert d.SLxImageAttributes.uiSequenceCount == len(self.images.keys())
                self.n_fields = d.SLxImageAttributes.uiSequenceCount
                self.dim = (d.SLxImageAttributes.uiHeight, d.SLxImageAttributes.uiWidth)
                assert d.SLxImageAttributes.uiBpcSignificant == 16
                assert d.SLxImageAttributes.uiBpcInMemory == 16
            elif block.name == "CustomData|X":
                self.x = self._data_block(block, self._f64, 8)
            elif block.name == "CustomData|Y":
                self.y = self._data_block(block, self._f64, 8)
            elif block.name == "CustomData|Z":
                self.z = self._data_block(block, self._f64, 8)
            elif block.name == "CustomData|PFS_STATUS":
                self.pfs_status = self._data_block(block, self._u32, 4)
            elif block.name == "CustomData|PFS_OFFSET":
                self.pfs_offset = self._data_block(block, self._u32, 4)
            elif block.name == "CustomData|Camera_ExposureTime1":
                self.exposure_time = self._data_block(block, self._f64, 8)
            elif block.name == "CustomData|CameraTemp1":
                self.camera_temp = self._data_block(block, self._f64, 8)

    def _hex(self, start, count=0x100):
        col = 16
        printable = [chr(i) if 32 <= i < 128 else "." for i in range(256)]
        rows = count // col
        for row in range(rows):
            row_data = self.data[start + row * col : start + (row + 1) * col]
            x = f"{start + row * 16:08X}"
            h = " ".join([f"{i:02X}" for i in row_data])
            a = "".join([printable[i] for i in row_data])
            print(f"{x}  {h}  {a}")
        print()

    def _u8(self, off):
        return struct.unpack("B", self.data[off : off + 1])[0]

    def _u16(self, off):
        return struct.unpack("H", self.data[off : off + 2])[0]

    def _u32(self, off):
        return struct.unpack("I", self.data[off : off + 4])[0]

    def _u64(self, off):
        return struct.unpack("L", self.data[off : off + 8])[0]

    def _f64(self, off):
        return struct.unpack("d", self.data[off : off + 8])[0]

    def _str16(self, off):
        start = off
        while True:
            if self._u16(off) == 0:
                break
            off += 2
        stop = off
        try:
            return str(self.data[start:stop], encoding="utf-16"), (stop - start) + 2
        except Exception as e:
            print(f"FAIL {start:08X} {stop:08X} len={stop-start+2}")
            self._hex(start, stop - start + 2)
            raise

    def _read_metadata_field(self, pos, indent=0):
        type_ = self._u8(pos)
        pos += 1

        strlen = self._u8(pos) * 2
        pos += 1

        key, _ = self._str16(pos)
        pos += strlen

        if type_ == 1:
            val = self._u8(pos)
            pos += 1
        elif type_ == 2 or type_ == 3:
            val = self._u32(pos)
            pos += 4
        elif type_ == 5:
            val = self._u64(pos)
            pos += 8
        elif type_ == 6:
            val = self._f64(pos)
            pos += 8
        elif type_ == 8:
            val, val_len = self._str16(pos)
            pos += val_len
        elif type_ == 9:
            val = "?"
            val_len = self._u64(pos)
            pos += 8 + val_len
        elif type_ == 11:
            dict_len, val = self._read_dict(pos, indent)
            pos += dict_len
        else:
            raise Exception(f"Unhandled type {type_}")

        return pos, key, val

    def _read_dict(self, pos, indent):
        """
        A dict is an int process_count of keys followed an (mysterious) length
        4-bytes process_count-of-keys
        8-bytes mysterious length
        followed by the keys values:
            1 byte-type
            1 byte key len
            n-bytes the key
            Value
                The value depends on the type which are all simple except dicts
                When the value is itself a dict then the mysterious length includes the offset
                of this dict in the outer dict. Why?
        """
        d = Munch()

        start = pos
        count_of_keys = self._u32(pos)
        pos += 4

        _ = self._u64(
            pos
        )  # This is weird, it seems to includes a running off plus the 12 bytes we just read
        pos += 8

        for ki in range(count_of_keys):
            pos, key, val = self._read_metadata_field(pos, indent + 2)
            d[key] = val

        # Weird padding
        pos += 8 * count_of_keys

        return pos - start, d

    def _metadata_section(self, block):
        pos = block.pos
        pos += 4  # Skip magic header

        name_len = self._u32(pos)
        pos += 4

        data_len = self._u64(pos)
        pos += 8

        pos += name_len

        end = pos + data_len

        d = Munch()
        while pos < end:
            pos, key, val = self._read_metadata_field(pos)
            d[key] = val

        return d

    def _data_block(self, block, func, elem_size):
        pos = block.pos

        pos += 4  # Skip magic header

        name_len = self._u32(pos)
        pos += 4

        data_len = self._u64(pos)
        pos += 8

        count = data_len // elem_size
        assert data_len % elem_size == 0
        data = [func(pos + name_len + i * elem_size) for i in range(count)]
        return data

    def _dumpd(self, d, indent=0):
        for k, v in d.items():
            if not isinstance(v, dict):
                print(f"{'  ' * indent}{k} = {v}")
            else:
                print(f"{'  ' * indent}{k} = DICT")
                self._dumpd(v, indent + 1)

    def get_fields(self, n_fields=None):
        """
        Returns numpy array of shape (n_fields, n_channels, dim, dim)
        """
        if n_fields is None:
            n_fields = self.n_fields
        ims = np.zeros((n_fields, self.n_channels, *self.dim))
        for field in range(n_fields):
            block = self.images[field]

            pos = block.pos
            pos += 4  # Skip magic header

            name_len = self._u32(pos)
            pos += 4

            data_len = self._u64(pos)
            pos += 8

            pos += name_len

            timestamp = self._f64(pos)
            pos += 8

            n_pixels = self.dim[0] * self.dim[1] * self.n_channels
            im = np.ndarray(
                (n_pixels,), buffer=self.data[pos : pos + n_pixels * 2], dtype="uint16"
            )
            for channel in range(self.n_channels):
                ims[field, channel, :, :] = np.reshape(
                    im[channel :: self.n_channels], self.dim
                )

        return ims

    def get_field(self, field, channel):
        block = self.images[field]

        pos = block.pos
        pos += 4  # Skip magic header

        name_len = self._u32(pos)
        pos += 4

        data_len = self._u64(pos)
        pos += 8

        pos += name_len

        timestamp = self._f64(pos)
        pos += 8

        n_pixels = self.dim[0] * self.dim[1] * self.n_channels
        im = np.ndarray(
            (n_pixels,), buffer=self.data[pos : pos + n_pixels * 2], dtype="uint16"
        )
        return np.reshape(im[channel :: self.n_channels], self.dim)
