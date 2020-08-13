#ifndef C_COMMON_H
#define C_COMMON_H


typedef __uint8_t Uint8;
typedef __uint32_t Uint32;
typedef __uint64_t Uint64;
typedef __uint128_t Uint128;

typedef __int8_t Sint8;
typedef __int32_t Sint32;
typedef __int64_t Sint64;
typedef __int128_t Sint128;

typedef float Float32;
typedef double Float64;

typedef Uint64 Size;
typedef Uint64 Index;
typedef Uint32 Size32;
typedef Uint32 Index32;
typedef Uint64 HashKey;
typedef Uint8 DyeType;
typedef Uint8 CycleKindType;
typedef Uint64 PIType;
typedef Float64 RecallType;
typedef Float32 RadType;
typedef Float32 Score;  // TODO: Rename ScoreType
typedef Float32 WeightType;


#define N_MAX_CHANNELS ((DyeType)(8))
#define NO_LABEL ((DyeType)(N_MAX_CHANNELS - 1))
#define N_MAX_CYCLES ((DyeType)64)
#define CYCLE_TYPE_PRE ((CycleKindType)(0))
#define CYCLE_TYPE_MOCK ((CycleKindType)(1))
#define CYCLE_TYPE_EDMAN ((CycleKindType)(2))
#define N_MAX_NEIGHBORS (8)


typedef struct {
    Uint8 *rows;
    Uint64 n_bytes_per_row;
    Uint64 n_max_rows;
    Uint64 n_rows;
    Uint64 readonly;
} Table;


typedef struct {
    Index dtr_i;
    Index pep_i;
    Size n_reads;
} DyePepRec;


Uint64 now();

// Ensure
void ensure(int expr, const char *fmt, ...);
#ifdef DEBUG
    #define ensure_only_in_debug ensure
#else
    #define ensure_only_in_debug(...) ((void)0)
#endif

// Trace
void _trace(const char *fmt, ...);
#ifdef DEBUG
    #define trace _trace
#else
    #define trace(...) ((void)0)
#endif

// Tables
Table table_init(Uint8 *base, Size n_bytes, Size n_bytes_per_row);
Table table_init_readonly(Uint8 *base, Size n_bytes, Size n_bytes_per_row);
Table table_init_subset(Table *src, Index row_i, Size n_rows, Uint64 is_readonly);

void *_table_get_row(Table *table, Index row);

#ifdef DEBUG
    #define table_get_row(table, row, type) (type *)_table_get_row(table, row)
#else
    #define table_get_row(table, row, type) (void *)(table->rows + table->n_bytes_per_row * row)
#endif

void table_set_row(Table *table, Index row_i, void *src);

Index table_add(Table *table, void *src, pthread_mutex_t *lock, char *table_name);

void table_validate(Table *table, void *ptr, char *msg);
#ifdef DEBUG
    #define table_validate_only_in_debug table_validate
#else
    #define table_validate_only_in_debug(...) ((void)0)
#endif

void table_dump(Table *table, char *msg);

typedef void (*ProgressFn)(int complete, int total, int retry);

#endif
