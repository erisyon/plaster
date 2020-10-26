#ifndef C_COMMON_H
#define C_COMMON_H


// This is now deprecated. It is in the process of being replaced by c_common.h

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

typedef float Float32;
typedef double Float64;

typedef Uint64 Bool;
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
typedef Float32 ScoreType;
typedef Float32 WeightType;
typedef Float32 IsolationType;
typedef Float32 RowKType;


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
    Index dyt_i;
    Index pep_i;
    Size n_reads;
} DyePepRec;


typedef struct {
    Index i;
    Size n;
} RLEBlock;

Uint64 now();

#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))


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
Table table_init(void *base, Size n_bytes, Size n_bytes_per_row);
Table table_init_readonly(void *base, Size n_bytes, Size n_bytes_per_row);
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

//Table rle_index_init(Index *src, Size n_src, Index *dst, Size n_dst);
//RLEBlock rle_index_get(Table *rle_table, Index pos);

typedef void (*ProgressFn)(int complete, int total, int retry);
typedef int (*CheckKeyboardInterruptFn)();


int sanity_check();

// hash
//----------------------------------------------------------------------------------------

typedef struct {
    HashKey key;
    union {
    	void *val;
    	IsolationType contention_val;
	};
} HashRec;


typedef struct {
    HashRec *recs;
    Uint64 n_max_recs;
    Uint64 n_active_recs;
} Hash;

Hash hash_init(HashRec *buffer, Size n_max_recs);
HashRec *hash_get(Hash hash, HashKey key);
void hash_dump(Hash hash);


// tab
//----------------------------------------------------------------------------------------

#define TAB_NO_LOCK (void *)0

#define TAB_NOT_GROWABLE (0)
#define TAB_GROWABLE (1<<1)
#define TAB_FLAGS_INT (1<<2)
#define TAB_FLAGS_FLOAT (1<<3)
#define TAB_FLAGS_UNSIGNED (1<<4)
#define TAB_FLAGS_HAS_ELEMS (1<<5)

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
Tab tab_subset(Tab *src, Uint64 row_i, Uint64 n_rows);
Tab tab_by_n_rows(void *base, Uint64 n_rows, Uint64 n_bytes_per_row, Uint64 flags);
Tab tab_by_size(void *base, Uint64 n_bytes, Uint64 n_bytes_per_row, Uint64 flags);
Tab tab_by_arr(void *base, Uint64 n_rows, Uint64 n_cols, Uint64 n_bytes_per_elem, Uint64 flags);
Tab tab_malloc_by_n_rows(Uint64 n_rows, Uint64 n_bytes_per_row, Uint64 flags);
Tab tab_malloc_by_size(Uint64 n_bytes, Uint64 n_bytes_per_row, Uint64 flags);
void tab_free(Tab *tab);
void *_tab_get(Tab *tab, Uint64 row_i, Uint64 flags, char *file, int line);
void _tab_set(Tab *tab, Uint64 row_i, void *src, char *file, int line);
Uint64 _tab_add(Tab *tab, void *src, pthread_mutex_t *lock, char *file, int line);
void _tab_validate(Tab *tab, void *ptr, char *file, int line);


#define tab_row(tab, row_i) _tab_get(tab, row_i, 0, __FILE__, __LINE__)
#define tab_var(typ, var, tab, row_i) typ *var = (typ *)_tab_get(tab, row_i, 0, __FILE__, __LINE__)
#define tab_ptr(typ, tab, row_i) (typ *)_tab_get(tab, row_i, 0, __FILE__, __LINE__)
#define tab_get(typ, tab, row_i) *(typ *)_tab_get(tab, row_i, 0, __FILE__, __LINE__)
#define tab_col(typ, tab, row_i, col_i) ((typ *)_tab_get(tab, row_i, TAB_FLAGS_HAS_ELEMS, __FILE__, __LINE__))[col_i]
#define tab_set(tab, row_i, src_ptr) _tab_set(tab, row_i, src_ptr, __FILE__, __LINE__)
#define tab_add(tab, src, lock) _tab_add(tab, src, lock, __FILE__, __LINE__)
#define tab_validate(tab, ptr) _tab_validate(tab, ptr, __FILE__, __LINE__)

#ifdef DEBUG
    #define tab_validate_only_in_debug(tab, ptr) _tab_validate(tab, ptr, __FILE__, __LINE__)
#else
    #define tab_validate_only_in_debug(...) ((void)0)
#endif

#define tab_alloca(table_name, n_rows, n_bytes_per_row) \
	void *buf##__LINE__ = (void *)alloca(n_rows * n_bytes_per_row); \
    memset(buf##__LINE__, 0, n_rows * n_bytes_per_row); \
    Tab table_name = tab_by_n_rows(buf##__LINE__, n_rows, n_bytes_per_row, TAB_NOT_GROWABLE)

#endif
