import numpy as np
import ctypes as c
from plaster.tools.schema import check


typedefs = {
    # typedef name, c type, python ctype
    "void": ("void", c.c_void_p),
    "Uint8": ("__uint8_t", c.c_ubyte),
    "Uint16": ("__uint16_t", c.c_ushort),
    "Uint32": ("__uint32_t", c.c_uint),
    "Uint64": ("__uint64_t", c.c_ulonglong),
    "Sint8": ("__int8_t", c.c_byte),
    "Sint16": ("__int16_t", c.c_short),
    "Sint32": ("__int32_t", c.c_int),
    "Sint64": ("__int64_t", c.c_longlong),
    "Float32": ("float", c.c_float),
    "Float64": ("double", c.c_double),
    "Bool": ("Uint64", c.c_ulonglong),
    "Size": ("Uint64", c.c_ulonglong),
    "Index": ("Uint64", c.c_ulonglong),
    "Size32": ("Uint32", c.c_uint),
    "Index32": ("Uint32", c.c_uint),
    "HashKey": ("Uint64", c.c_ulonglong),
    "DyeType": ("Uint8", c.c_ubyte),
    "CycleKindType": ("Uint8", c.c_ubyte),
    "PIType": ("Uint64", c.c_ulonglong),
    "RecallType": ("Float64", c.c_float),
    "RadType": ("Float32", c.c_float),
    "ScoreType": ("Float32", c.c_float),
    "WeightType": ("Float32", c.c_float),
    "IsolationType": ("Float32", c.c_float),
    "RowKType": ("Float32", c.c_float),
}


def typedef_to_ctype(typ):
    return typedefs[typ][1]


class Tab(c.Structure):
    _fields_ = [
        ("base", c.c_void_p),
        ("n_bytes_per_row", c.c_ulonglong),
        ("n_max_rows", c.c_ulonglong),
        ("n_rows", c.c_ulonglong),
        ("b_growable", c.c_ulonglong),
    ]

    @classmethod
    def from_mat(cls, mat, expected_dtype):
        check.array_t(mat, ndim=2, dtype=expected_dtype)
        tab = Tab()
        tab.base = mat.ctypes.data_as(c.c_void_p)
        tab.n_bytes_per_row = mat.itemsize * mat.shape[1]
        tab.n_max_rows = mat.shape[0]
        tab.n_rows = mat.shape[0]
        tab.b_growable = 0
        return tab


class FixupStructure(c.Structure):
    @classmethod
    def struct_fixup(cls):
        """
        Converts the "_fixup_fields" element of a x.Structure sub-class
        to create a proper ctypes Structure.
        Uses the typedefs above.

        Example:

            class MyStruct(c.Structure):
                _fixup_fields = [
                    ("n_elems", "Size"),
                    ("elems", "Float64 *"),
                ]

            struct_fixup(MyStruct)

        """
        setattr(cls, "_tab_types", dict())

        fields = []
        for f in cls._fixup_fields:
            field_name = f[0]

            if isinstance(f[1], str):
                typedef_parts = f[1].split()

                n_parts = len(typedef_parts)
                if n_parts == 1:
                    ctypes_type = typedefs[typedef_parts[0]][1]
                    fields += [(field_name, ctypes_type)]
                elif n_parts > 1 and typedef_parts[-1] == "*":
                    ctypes_type = typedefs.get(typedef_parts[0])
                    if ctypes_type is not None:
                        ctypes_type = ctypes_type[1]
                        fields += [(field_name, c.POINTER(ctypes_type))]
                    else:
                        fields += [(field_name, c.POINTER(c.c_void_p))]
                else:
                    raise TypeError(
                        f"Unknown type reference '{field_name}, {typedef_parts}'"
                    )

            elif f[1] is Tab:
                assert len(f) == 3
                fields += [(field_name, Tab)]
                cls._tab_types[field_name] = f[2]

            else:
                fields += [(field_name, f[1])]

        cls._fields_ = fields

    @classmethod
    def struct_emit_header(cls, header_fp):
        """
        Emits a header file for the struct_klass to header_fp

        Example:

            class MyStruct(c.Structure):
                _fixup_fields = [
                    ("n_elems", "Size"),
                    ("elems", "Float64 *"),
                ]

            with open("foo.h", "w") as foo_header_fp:
                struct_emit_header(MyStruct, foo_header_fp)


            Writes to foo.h as:

            typedef __uint8_t Uint8;
            // other typedefs here...
            typedef struct {
                Size n_elems;
                Float64 *elems;
            } MyStruct;

        """
        print("typedef struct {", file=header_fp)
        for f in cls._fixup_fields:
            if isinstance(f[1], str):
                typename = f[1]
            else:
                typename = f[1].__name__
            print(f"    {typename} {f[0]};", file=header_fp)
        print("} " + f"{cls.__name__};", file=header_fp)

    @classmethod
    def tab_type(cls, field):
        return cls._tab_types[field]
