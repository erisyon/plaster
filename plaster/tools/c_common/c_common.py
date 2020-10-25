import numpy as np
import ctypes as c
from plaster.tools.schema import check


typedefs = {
    "Uint8": ("__uint8_t", 1, c.c_ubyte, False),
    "Uint16": ("__uint16_t", 2, c.c_ushort, False),
    "Uint32": ("__uint32_t", 4, c.c_uint, False),
    "Uint64": ("__uint64_t", 8, c.c_ulonglong, False),
    "Sint8": ("__int8_t", 1, c.c_byte, False),
    "Sint16": ("__int16_t", 2, c.c_short, False),
    "Sint32": ("__int32_t", 4, c.c_int, False),
    "Sint64": ("__int64_t", 8, c.c_longlong, False),
    "Float32": ("float", 4, c.c_float, False),
    "Float64": ("double", 8, c.c_double, False),
    "Bool": ("Uint64", 8, c.c_ulonglong, True),
    "Size": ("Uint64", 8, c.c_ulonglong, True),
    "Index": ("Uint64", 8, c.c_ulonglong, True),
    "Size32": ("Uint32", 4, c.c_uint, True),
    "Index32": ("Uint32", 4, c.c_uint, True),
    "HashKey": ("Uint64", 8, c.c_ulonglong, True),
    "DyeType": ("Uint8", 1, c.c_ubyte, True),
    "CycleKindType": ("Uint8", 1, c.c_ubyte, True),
    "PIType": ("Uint64", 8, c.c_ulonglong, True),
    "RecallType": ("Float64", 8, c.c_float, True),
    "RadType": ("Float32", 4, c.c_float, True),
    "ScoreType": ("Float32", 4, c.c_float, True),
    "WeightType": ("Float32", 4, c.c_float, True),
    "IsolationType": ("Float32", 4, c.c_float, True),
    "RowKType": ("Float32", 4, c.c_float, True),
}


def typedef_to_ctype(typ):
    return typedefs[typ][2]


def typedefs_emit(fp):
    for t, info in typedefs.items():
        if info[3]:
            print(f"typedef {info[0]} {t};", file=fp)


def typedefs_sanity_check_emit(fp):
    print("void sanity_check() {", file=fp)
    for t, info in typedefs.items():
        print(f"assert(sizeof({t}) == {info[1]});", file=fp)
    print("}", file=fp)


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
                    ctypes_type = typedefs[typedef_parts[0]][2]
                    fields += [(field_name, ctypes_type)]
                elif n_parts == 2 and typedef_parts[1] == "*":
                    ctypes_type = typedefs[typedef_parts[1]][2]
                    fields += [(field_name, c.POINTER(ctypes_type))]
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
                typedefs_emit(foo_header_fp)
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
                typename = f[1].split()[0]  # Split to separate Tab for type
            else:
                typename = f[1].__name__
            print(f"    {typename} {f[0]};", file=header_fp)
        print("} " + f"{cls.__name__};", file=header_fp)

    @classmethod
    def tab_type(cls, field):
        return cls._tab_types[field]
