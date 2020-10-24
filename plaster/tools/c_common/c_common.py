import ctypes as c

typedefs = {
    "Uint8": ("__uint8_t", 1, c.c_ubyte),
    "Uint16": ("__uint16_t", 2, c.c_ushort),
    "Uint32": ("__uint32_t", 4, c.c_uint),
    "Uint64": ("__uint64_t", 8, c.c_ulonglong),
    "Sint8": ("__int8_t", 1, c.c_byte),
    "Sint16": ("__int16_t", 2, c.c_short),
    "Sint32": ("__int32_t", 4, c.c_int),
    "Sint64": ("__int64_t", 8, c.c_longlong),
    "Float32": ("float", 4, c.c_float),
    "Float64": ("double", 8, c.c_double),
    "Bool": ("Uint64", 8, c.c_ulonglong),
    "Size": ("Uint64", 8, c.c_ulonglong),
    "Index": ("Uint64", 8, c.c_ulonglong),
    "Size32": ("Uint32", 4, c.c_uint),
    "Index32": ("Uint32", 4, c.c_uint),
    "HashKey": ("Uint64", 8, c.c_ulonglong),
    "DyeType": ("Uint8", 1, c.c_ubyte),
    "CycleKindType": ("Uint8", 1, c.c_ubyte),
    "PIType": ("Uint64", 8, c.c_ulonglong),
    "RecallType": ("Float64", 8, c.c_float),
    "RadType": ("Float32", 4, c.c_float),
    "ScoreType": ("Float32", 4, c.c_float),
    "WeightType": ("Float32", 4, c.c_float),
    "IsolationType": ("Float32", 4, c.c_float),
    "RowKType": ("Float32", 4, c.c_float),
}


def typedef_to_ctype(typ):
    return typedefs[typ][2]


def typedefs_emit(fp):
    for t, info in typedefs.items():
        print(f"typedef {t} {info[0]};", file=fp)


def typedefs_sanity_check_emit(fp):
    print("void sanity_check() {", file=fp)
    for t, info in typedefs.items():
        print(f"assert(sizeof({t}) == {info[1]});", file=fp)
    print("}", file=fp)


def struct_fixup(struct_klass):
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
    fields = []
    for f in struct_klass._fixup_fields:
        field_name = f[0]
        typedef_parts = f[1].split()

        if len(typedef_parts) == 2 and typedef_parts[1] == "*":
            ctypes_type = typedefs[typedef_parts[1]][2]
            fields += [(field_name, c.POINTER(ctypes_type))]
        elif len(typedef_parts) == 1:
            ctypes_type = typedefs[typedef_parts[0]][2]
            fields += [(field_name, ctypes_type)]
        else:
            raise TypeError(f"Unknown type reference '{field_name}, {typedef_parts}'")

    struct_klass._fields_ = fields


def struct_emit_header(struct_klass, header_fp):
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
    for f in struct_klass._fixup_fields:
        print(f"    {f[1]} {f[0]};", file=header_fp)
    print("} " + f"{struct_klass.__name__};", file=header_fp)
