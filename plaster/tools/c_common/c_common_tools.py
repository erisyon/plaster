import ctypes as c

import numpy as np
from plaster.tools.schema import check


class CException(Exception):
    def __init__(self, s):
        super().__init__(s.decode("ascii"))


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
    "RecallType": ("Float64", c.c_double),
    "RadType": ("Float32", c.c_float),
    "ScoreType": ("Float32", c.c_float),
    "WeightType": ("Float32", c.c_float),
    "IsolationType": ("Float32", c.c_float),
    "RowKType": ("Float32", c.c_float),
    "ProgressFn": ("ProgressFn", c.CFUNCTYPE(c.c_void_p, c.c_int, c.c_int, c.c_int)),
    "KeyboardInterruptFn": ("KeyboardInterruptFn", c.CFUNCTYPE(c.c_int)),
}


def typedef_to_ctype(typ):
    return typedefs[typ][1]


class Tab(c.Structure):
    # See c_common.h for duplicate defines
    TAB_NOT_GROWABLE = 0
    TAB_GROWABLE = 1 << 0
    TAB_FLAGS_INT = 1 << 1
    TAB_FLAGS_FLOAT = 1 << 2
    TAB_FLAGS_UNSIGNED = 1 << 3
    TAB_FLAGS_HAS_ELEMS = 1 << 4

    _fields_ = [
        ("base", c.c_void_p),
        ("n_bytes_per_row", c.c_ulonglong),
        ("n_max_rows", c.c_ulonglong),
        ("n_rows", c.c_ulonglong),
        ("n_cols", c.c_ulonglong),
        ("n_bytes_per_elem", c.c_ulonglong),
        ("flags", c.c_ulonglong),
    ]

    @classmethod
    def from_mat(cls, mat, expected_dtype):
        if mat is None:
            # It is allowed to pass empty table
            # (tables that will not be used in certain modes)
            tab = Tab()
            tab.base = 0
            tab.n_bytes_per_row = 0
            tab.n_max_rows = 0
            tab.n_rows = 0
            tab.n_cols = 0
            tab.n_bytes_per_elem = 0
            tab.flags = 0

        else:
            check.array_t(mat, dtype=expected_dtype, c_contiguous=True)
            tab = Tab()
            tab.base = mat.ctypes.data_as(c.c_void_p)
            if mat.ndim == 2:
                tab.n_bytes_per_row = mat.itemsize * mat.shape[1]
                tab.n_max_rows = mat.shape[0]
                tab.n_rows = mat.shape[0]
                tab.n_cols = mat.shape[1]
            elif mat.ndim == 1:
                tab.n_bytes_per_row = mat.itemsize
                tab.n_max_rows = mat.shape[0]
                tab.n_rows = mat.shape[0]
                tab.n_cols = 1
            else:
                raise Exception("Unsupported dimension for mat")
            tab.n_bytes_per_elem = mat.itemsize
            tab.flags = Tab.TAB_FLAGS_HAS_ELEMS

            if np.issubdtype(mat.dtype, np.integer):
                tab.flags |= Tab.TAB_FLAGS_INT
            else:
                tab.flags |= Tab.TAB_FLAGS_FLOAT

            # TODO: Figure out how to check for unsigned dtypes

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
        if getattr(cls, "_fixed_up", False):
            return

        setattr(cls, "_fixed_up", True)
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

            elif f[1] is F64Arr:
                fields += [(field_name, F64Arr)]
                cls._tab_types[field_name] = c.POINTER(c.c_double)

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


class F64Arr(c.Structure):
    # See c_common.h for duplicate define
    MAX_ARRAY_DIMS = 4

    _fields_ = [
        ("base", c.c_void_p),
        ("n_dims", c.c_ulonglong),
        ("shape0", c.c_ulonglong),
        ("shape1", c.c_ulonglong),
        ("shape2", c.c_ulonglong),
        ("shape3", c.c_ulonglong),
        ("pitch0", c.c_ulonglong),
        ("pitch1", c.c_ulonglong),
        ("pitch2", c.c_ulonglong),
        ("pitch3", c.c_ulonglong),
    ]

    @classmethod
    def from_ndarray(cls, ndarr):
        check.array_t(ndarr, dtype=np.float64, c_contiguous=True)
        arr = F64Arr()
        arr.base = ndarr.ctypes.data_as(c.c_void_p)

        assert ndarr.ndim <= cls.MAX_ARRAY_DIMS
        arr.n_dims = ndarr.ndim

        arr.shape0 = ndarr.shape[0] if ndarr.ndim >= 1 else 0
        arr.shape1 = ndarr.shape[1] if ndarr.ndim >= 2 else 0
        arr.shape2 = ndarr.shape[2] if ndarr.ndim >= 3 else 0
        arr.shape3 = ndarr.shape[3] if ndarr.ndim >= 4 else 0

        # Pitch is the cumulative size of the blocks less than each dimension
        # (ie how much has to be added to advance one index in eah dimension)
        arr.pitch3 = 1 if ndarr.ndim >= 4 else 1
        arr.pitch2 = max(1, arr.shape3) * arr.pitch3 if ndarr.ndim >= 3 else 1
        arr.pitch1 = max(1, arr.shape2) * arr.pitch2 if ndarr.ndim >= 2 else 1
        arr.pitch0 = max(1, arr.shape1) * arr.pitch1 if ndarr.ndim >= 1 else 1

        return arr


class HashVal(c.Union):
    _fields_ = [("val", c.c_void_p), ("contention_val", c.c_float)]


class HashRec(c.Structure):
    _fields_ = [("key", c.c_uint64), ("val_union", HashVal)]


class Hash(c.Structure):
    _fields_ = [
        ("recs", c.POINTER(HashRec)),
        ("n_max_recs", c.c_uint64),
        ("n_active_recs", c.c_uint64),
    ]
