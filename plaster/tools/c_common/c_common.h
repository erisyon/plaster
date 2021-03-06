#ifndef C_COMMON_H
#define C_COMMON_H

// Base Sized-types
typedef __uint8_t Uint8;
typedef __uint16_t Uint16;
typedef __uint32_t Uint32;
typedef __uint64_t Uint64;
typedef __uint128_t Uint128;
typedef __int8_t Sint8;
typedef __int16_t Sint16;
typedef __int32_t Sint32;
typedef __int64_t Sint64;
typedef __int128_t Sint128;
typedef __uint64_t Bool;
typedef float Float32;
typedef double Float64;

// Types for plaster
typedef Uint64 Size;
typedef Uint64 Index;
typedef Uint32 Size32;
typedef Uint32 Index32;
typedef Uint64 HashKey;
typedef Uint8 DyeType;
typedef Uint8 CycleKindType;
typedef Uint64 PIType;
typedef Uint64 DyePepType;
typedef Float64 RecallType;
typedef Float32 RadType;
typedef Float32 ScoreType;
typedef Uint64 DytWeightType;
typedef Float32 IsolationType;
typedef Float64 RowKType;
typedef Uint64 DytIndexType;

typedef struct {
    Index dyt_i;
    Index pep_i;
    Size n_reads;
} DyePepRec;

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

// Used for returning exception-like values from calls
#define check_and_return(expr, static_fail_string)                                                                     \
    if(!(expr))                                                                                                        \
        return static_fail_string;

// Ensure
void ensure(int expr, const char *fmt, ...);
#ifdef DEBUG
    #define ensure_only_in_debug ensure
#else
    #define ensure_only_in_debug(...) ((void)0)
#endif

// Trace
extern FILE *_log;
void _trace(char *file, int line, const char *fmt, ...);
#ifdef DEBUG
    #define trace(...) _trace(__FILE__, __LINE__, __VA_ARGS__)
#else
    #define trace(...) ((void)0)
#endif

typedef void (*ProgressFn)(int complete, int total, int retry);
typedef int (*KeyboardInterruptFn)();

#define N_MAX_CHANNELS ((DyeType)(8))
#define NO_LABEL ((DyeType)(N_MAX_CHANNELS - 1))
#define N_MAX_CYCLES ((DyeType)64)
#define CYCLE_TYPE_PRE ((CycleKindType)(0))
#define CYCLE_TYPE_MOCK ((CycleKindType)(1))
#define CYCLE_TYPE_EDMAN ((CycleKindType)(2))
#define N_MAX_NEIGHBORS (8)

Uint64 now();

// Hash
//----------------------------------------------------------------------------------------

typedef Uint64 HashKey;

typedef struct {
    HashKey key;
    union {
        void *val;
        float contention_val;
    };
} HashRec;

typedef struct {
    HashRec *recs;
    Uint64 n_max_recs;
    Uint64 n_active_recs;
} Hash;

Hash hash_init(HashRec *buffer, Uint64 n_max_recs);
HashRec *hash_get(Hash hash, HashKey key);
void hash_dump(Hash hash);

// Tab
//----------------------------------------------------------------------------------------

#define TAB_NO_LOCK (void *)0

// See c_common_tools.py for duplicate defines
#define TAB_NOT_GROWABLE (0)
#define TAB_GROWABLE (1 << 1)
#define TAB_FLAGS_INT (1 << 2)
#define TAB_FLAGS_FLOAT (1 << 3)
#define TAB_FLAGS_UNSIGNED (1 << 4)
#define TAB_FLAGS_HAS_ELEMS (1 << 5)

typedef struct {
    void *base;
    Uint64 n_bytes_per_row;
    Uint64 n_max_rows;
    Uint64 n_rows;
    Uint64 n_cols; // Only applies if all columns are the same size
    Uint64 n_bytes_per_elem;
    Uint64 flags;
} Tab;

void tab_tests();
void tab_dump(Tab *tab, char *msg);
Tab _tab_subset(Tab *src, Uint64 row_i, Uint64 n_rows, char *file, int line);
Tab tab_by_n_rows(void *base, Uint64 n_rows, Uint64 n_bytes_per_row, Uint64 flags);
Tab tab_by_size(void *base, Uint64 n_bytes, Uint64 n_bytes_per_row, Uint64 flags);
Tab tab_by_arr(void *base, Uint64 n_rows, Uint64 n_cols, Uint64 n_bytes_per_elem, Uint64 flags);
Tab tab_malloc_by_n_rows(Uint64 n_rows, Uint64 n_bytes_per_row, Uint64 flags);
Tab tab_malloc_by_size(Uint64 n_bytes, Uint64 n_bytes_per_row, Uint64 flags);
void tab_free(Tab *tab);
void *_tab_get(Tab *tab, Uint64 row_i, Uint64 flags, char *file, int line);
void _tab_set(Tab *tab, Uint64 row_i, void *src, char *file, int line);
void _tab_set_col(Tab *tab, Uint64 row_i, Uint64 col_i, void *src, char *file, int line);
Uint64 _tab_add(Tab *tab, void *src, pthread_mutex_t *lock, char *file, int line);
void _tab_validate(Tab *tab, void *ptr, char *file, int line);

#define tab_row(tab, row_i) _tab_get(tab, row_i, 0, __FILE__, __LINE__)
#define tab_var(typ, var, tab, row_i) typ *var = (typ *)_tab_get(tab, row_i, 0, __FILE__, __LINE__)
#define tab_ptr(typ, tab, row_i) (typ *)_tab_get(tab, row_i, 0, __FILE__, __LINE__)
#define tab_get(typ, tab, row_i) *(typ *)_tab_get(tab, row_i, 0, __FILE__, __LINE__)
#define tab_col(typ, tab, row_i, col_i) ((typ *)_tab_get(tab, row_i, TAB_FLAGS_HAS_ELEMS, __FILE__, __LINE__))[col_i]
#define tab_set(tab, row_i, src_ptr) _tab_set(tab, row_i, src_ptr, __FILE__, __LINE__)
#define tab_set_col(tab, row_i, col_i, src_ptr) _tab_set_col(tab, row_i, col_i, src_ptr, __FILE__, __LINE__)
#define tab_add(tab, src, lock) _tab_add(tab, src, lock, __FILE__, __LINE__)
#define tab_validate(tab, ptr) _tab_validate(tab, ptr, __FILE__, __LINE__)
#define tab_subset(src, row_i, n_rows) _tab_subset(src, row_i, n_rows, __FILE__, __LINE__)

#ifdef DEBUG
    #define tab_validate_only_in_debug(tab, ptr) _tab_validate(tab, ptr, __FILE__, __LINE__)
#else
    #define tab_validate_only_in_debug(...) ((void)0)
#endif

#define tab_alloca(table_name, n_rows, n_bytes_per_row)                                                                \
    void *buf##__LINE__ = (void *)alloca(n_rows * n_bytes_per_row);                                                    \
    memset(buf##__LINE__, 0, n_rows *n_bytes_per_row);                                                                 \
    Tab table_name = tab_by_n_rows(buf##__LINE__, n_rows, n_bytes_per_row, TAB_NOT_GROWABLE)

// F64Arr
//----------------------------------------------------------------------------------------

#define MAX_ARRAY_DIMS (4)

typedef struct {
    Float64 *base;
    Size n_dims;

    // Shape is the number of elements in each dimensions
    // or zero if none.
    Size shape[MAX_ARRAY_DIMS];

    // pitch is the product of all subordinate shapes
    // (ie, the amount you need to add to an index of that
    // dimension to get to the next element).
    Size pitch[MAX_ARRAY_DIMS];
} F64Arr;

void f64arr_set_shape(F64Arr *arr, Size n_dims, Size *shape);
F64Arr f64arr(void *base, Size n_dims, Size *shape);
F64Arr f64arr_subset(F64Arr *src, Index i);
F64Arr f64arr_malloc(Size n_dims, Size *shape);
void f64arr_free(F64Arr *arr);

Float64 *f64arr_ptr1(F64Arr *arr, Index i);
Float64 *f64arr_ptr2(F64Arr *arr, Index i, Index j);
Float64 *f64arr_ptr3(F64Arr *arr, Index i, Index j, Index k);
Float64 *f64arr_ptr4(F64Arr *arr, Index i, Index j, Index k, Index l);

#define in_bounds(x, a, b) (((a) <= (x)) && ((x) < (b)))

#endif
