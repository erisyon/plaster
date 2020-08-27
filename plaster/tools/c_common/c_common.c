#include "math.h"
#include "stdint.h"
#include "alloca.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdarg.h"
#include "memory.h"
#include "unistd.h"
#include "inttypes.h"
#include "time.h"
#include "pthread.h"
#include "c_common.h"


Uint64 now() {
    struct timespec spec;
    clock_gettime(CLOCK_MONOTONIC_RAW, &spec);
    return (spec.tv_sec) * 1000000 + spec.tv_nsec / 1000;
}


void ensure(int expr, const char *fmt, ...) {
    // Replacement for assert with var-args and local control of compilation.
    // See ensure_only_in_debug below.
    va_list args;
    va_start(args, fmt);
    if(!expr) {
        vfprintf(stderr, fmt, args);
        fprintf(stderr, "\n");
        fflush(stderr);
        exit(1);
    }
    va_end(args);
}


void _trace(const char *fmt, ...) {
    // Replacement for printf that also flushes
    va_list args;
    va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    fflush(stdout);
    va_end(args);
}


// Table
//=========================================================================================

Table table_init(void *base, Size n_bytes, Size n_bytes_per_row) {
    // Wrap an externally allocated memory buffer ptr as a Table.
    // TODO: Put these checks by in optionally with a param (maybe add a table name too)
    // ensure(n_bytes_per_row % 4 == 0, "Mis-aligned table row size");
    // ensure((Uint64)base % 8 == 0, "Mis-aligned table");
    Table table;
    table.rows = (Uint8 *)base;
    table.n_rows = 0;
    table.n_bytes_per_row = n_bytes_per_row;
    table.n_max_rows = n_bytes / n_bytes_per_row;
    table.readonly = 0;
    memset(table.rows, 0, n_bytes);
    return table;
}


Table table_init_readonly(void *base, Size n_bytes, Size n_bytes_per_row) {
    // Wrap an externally allocated memory buffer ptr as a Table.
    // TODO: Put these checks by in optionally with a param (maybe add a table name too)
    // ensure(n_bytes_per_row % 4 == 0, "Mis-aligned table row size");
    // ensure((Uint64)base % 8 == 0, "Mis-aligned table");
    Table table;
    table.rows = (Uint8 *)base;
    table.n_bytes_per_row = n_bytes_per_row;
    table.n_max_rows = n_bytes / n_bytes_per_row;
    table.n_rows = table.n_max_rows;
    return table;
}


Table table_init_subset(Table *src, Index row_i, Size n_rows, Uint64 is_readonly) {
    Index last_row = row_i + n_rows;
    last_row = last_row < src->n_max_rows ? last_row : src->n_max_rows;
    n_rows = last_row - row_i;
    ensure(n_rows >= 0, "table_init_subset has illegal size");

    Table table;
    table.rows = (Uint8 *)(src->rows + src->n_bytes_per_row * row_i);
    table.n_bytes_per_row = src->n_bytes_per_row;
    table.n_max_rows = n_rows;
    table.n_rows = is_readonly ? n_rows : 0;
    table.readonly = is_readonly;
    return table;
}


void *_table_get_row(Table *table, Index row) {
    // Fetch a row from a table with bounds checking if activated
    ensure_only_in_debug(0 <= row && row < table->n_rows, "table get outside bounds");
    return (void *)(table->rows + table->n_bytes_per_row * row);
}


Index table_add(Table *table, void *src, pthread_mutex_t *lock, char *table_name) {
    // Add a row to the table and halt on overflow.
    // Optionally copies src into place if it isn't NULL.
    // Returns the row_i where the data was written (or will be written)
    // This is a potential race condition; a mutex may be warranted.  TODO: Test
    if(lock) pthread_mutex_lock(lock);
    Index row_i = table->n_rows;
    table->n_rows ++;
    if(lock) pthread_mutex_unlock(lock);
    ensure_only_in_debug(!table->readonly, "Attempting to write to a readonly table");
    ensure(row_i < table->n_max_rows, "Table overflow on %s. max_rows=%ld", table_name, table->n_max_rows);
    if(src != 0) {
        memcpy(table->rows + table->n_bytes_per_row * row_i, src, table->n_bytes_per_row);
    }
    return row_i;
}


void table_set_row(Table *table, Index row_i, void *src) {
    // Set a row to the table and halt on overflow.
    ensure_only_in_debug(!table->readonly, "Attempting to set to a readonly table");
    ensure(0 <= row_i && row_i < table->n_max_rows, "Table overflow");
    if(src != 0) {
        memcpy(table->rows + table->n_bytes_per_row * row_i, src, table->n_bytes_per_row);
    }
}


void table_validate(Table *table, void *ptr, char *msg) {
    // Check that a ptr is valid on a table
    // Use table_validate_only_in_debug (see below)
    Sint64 byte_offset = ((Uint8 *)ptr - table->rows);
    ensure(byte_offset % table->n_bytes_per_row == 0, msg);
    ensure(0 <= byte_offset && (Size)byte_offset < table->n_bytes_per_row * table->n_max_rows, msg);
}


void table_dump(Table *table, char *msg) {
    printf("table %s\n", msg);
    printf("rows=%p\n", table->rows);
    printf("n_bytes_per_row=%ld\n", table->n_bytes_per_row);
    printf("n_max_rows=%ld\n", table->n_max_rows);
    printf("n_rows=%ld\n", table->n_rows);
    printf("readonly=%ld\n", table->readonly);
}


// Tab
//=========================================================================================


Tab tab_by_size(void *base, Size n_bytes, Size n_bytes_per_row, int b_growable) {
    Tab tab;
    tab.base = base;
    tab.n_bytes_per_row = n_bytes_per_row;
    tab.n_max_rows = n_bytes / n_bytes_per_row;
    tab.b_growable = b_growable;
    if(b_growable) {
        memset(tab.base, 0, n_bytes);
        tab.n_rows = 0;
    }
    else {
        tab.n_rows = tab.n_max_rows;
    }
    return tab;
}


Tab tab_by_n_rows(void *base, Size n_rows, Size n_bytes_per_row, int b_growable) {
    return tab_by_size(base, n_rows * n_bytes_per_row, n_bytes_per_row, b_growable);
}


Tab tab_subset(Tab *src, Index row_i, Size n_rows) {
    ensure_only_in_debug(n_rows >= 0, "tab_subset has illegal n_rows %lu", n_rows);
    ensure_only_in_debug(0 <= row_i && row_i < src->n_rows, "tab_subset has illegal row_i %lu", row_i);

    Index last_row = row_i + n_rows;
    last_row = last_row < src->n_max_rows ? last_row : src->n_max_rows;
    n_rows = last_row - row_i;

    Tab tab;
    tab.base = src->base + src->n_bytes_per_row * row_i;
    tab.n_bytes_per_row = src->n_bytes_per_row;
    tab.n_max_rows = n_rows;
    tab.n_rows = n_rows;
    tab.b_growable = TAB_NOT_GROWABLE;
    return tab;
}


void *_tab_get(Tab *tab, Index row_i, char *file, int line) {
    ensure_only_in_debug(0 <= row_i && row_i < tab->n_rows, "tab_get outside bounds @%s:%d requested=%lu n_rows=%lu n_bytes_per_row=%lu", file, line, row_i, tab->n_rows, tab->n_bytes_per_row);
    return (void *)(tab->base + tab->n_bytes_per_row * row_i);
}


void _tab_set(Tab *tab, Index row_i, void *src, char *file, int line) {
    ensure_only_in_debug(0 <= row_i && row_i < tab->n_rows, "tab_set outside bounds @%s:%d row_i=%lu n_rows=%lu n_bytes_per_row=%lu", file, line, row_i, tab->n_rows, tab->n_bytes_per_row);
    if(src != (void *)0) {
        memcpy(tab->base + tab->n_bytes_per_row * row_i, src, tab->n_bytes_per_row);
    }
}


Index _tab_add(Tab *tab, void *src, pthread_mutex_t *lock, char *file, int line) {
    ensure_only_in_debug(tab->b_growable, "Attempting to grow to a un-growable table @%s:%d", file, line);
    if(lock) pthread_mutex_lock(lock);
    Index row_i = tab->n_rows;
    tab->n_rows ++;
    if(lock) pthread_mutex_unlock(lock);
    ensure_only_in_debug(0 <= row_i && row_i < tab->n_max_rows, "Table overflow @%s:%d. n_max_rows=%lu", file, line, tab->n_max_rows);
    if(src != (void *)0) {
        memcpy(tab->base + tab->n_bytes_per_row * row_i, src, tab->n_bytes_per_row);
    }
    return row_i;
}


void _tab_validate(Tab *tab, void *ptr, char *file, int line) {
    // Check that a ptr is valid on a table
    ensure_only_in_debug(
        tab->base <= ptr && ptr < tab->base + tab->n_bytes_per_row * tab->n_rows,
        "Tab ptr invalid @%s:%d. n_max_rows=%lu", file, line, tab->n_max_rows
    );
}

void tab_dump(Tab *tab, char *msg) {
    printf("table %s:\n", msg);
    printf("  base=%p\n", tab->base);
    printf("  n_bytes_per_row=%ld\n", tab->n_bytes_per_row);
    printf("  n_max_rows=%ld\n", tab->n_max_rows);
    printf("  n_rows=%ld\n", tab->n_rows);
    printf("  b_growable=%d\n", tab->b_growable);
}


void tab_tests() {
    #define N_ROWS (5)
    #define N_COLS (3)

    int buf[N_ROWS * N_COLS];
    for(int r=0; r<N_ROWS; r++) {
        for(int c=0; c<N_COLS; c++) {
            buf[r * N_COLS + c] = r<<16 | c;
        }
    }

    Tab tab_a = tab_by_n_rows(buf, N_ROWS * N_COLS, N_COLS * sizeof(int), TAB_NOT_GROWABLE);
    ensure(*(int *)tab_row(&tab_a, 0) == (0), "tab_row[0,0] wrong");
    ensure(*(int *)tab_row(&tab_a, 1) == (1<<16 | 0), "tab_row[1,1] wrong");
    ensure(((int *)tab_row(&tab_a, 1))[1] == (1<<16 | 1), "tab_row[1,1] wrong");

    tab_var(int, b, &tab_a, 1);
    ensure(b[0] == (1<<16 | 0), "tab_var[1,0] wrong");
    ensure(b[1] == (1<<16 | 1), "tab_var[1,1] wrong");

    tab_var(int, row_1, &tab_a, 1);
    tab_set(&tab_a, 4, row_1);
    tab_var(int, row_4, &tab_a, 1);
    ensure(row_4[0] == (1<<16 | 0), "tab_var[4,0] wrong after copy");
    ensure(row_4[2] == (1<<16 | 2), "tab_var[4,2] wrong after copy");

    // RESET to zeros
    memset(buf, 0, N_ROWS * N_COLS * sizeof(int));

    Tab tab_b = tab_by_n_rows(buf, N_ROWS, N_COLS * sizeof(int), TAB_GROWABLE);
    ensure(tab_b.n_rows == 0, "n_rows wrong after reset");
    ensure(tab_b.n_max_rows == N_ROWS, "n_max_rows wrong after reset");

    int row[N_COLS] = { 0, 1, 2 };
    tab_add(&tab_b, row, TAB_NO_LOCK);
    tab_var(int, c, &tab_b, 0);
    ensure(c[0] == 0, "c[0] wrong");
    ensure(c[1] == 1, "c[1] wrong");
    ensure(c[2] == 2, "c[2] wrong");
    ensure(tab_b.n_rows == 1, "tab_b.n_rows wrong");

    // RESET to original
    for(int r=0; r<N_ROWS; r++) {
        for(int c=0; c<N_COLS; c++) {
            buf[r * N_COLS + c] = r<<16 | c;
        }
    }
    Tab tab_s = tab_subset(&tab_a, 2, 2);
    tab_var(int, s, &tab_s, 0);
    ensure(s[0] == (2<<16 | 0), "s[0] wrong");
    ensure(s[1] == (2<<16 | 1), "s[0] wrong");
    ensure(s[2] == (2<<16 | 2), "s[0] wrong");
    s = tab_ptr(int, &tab_s, 1);
    ensure(s[0] == (3<<16 | 0), "s[1] wrong");
    ensure(s[1] == (3<<16 | 1), "s[1] wrong");
    ensure(s[2] == (3<<16 | 2), "s[1] wrong");
}


int sanity_check() {
    ensure(sizeof(Uint8) == 1, "Wrong size: Uint8");
    ensure(sizeof(Uint16) == 2, "Wrong size: Uint16");
    ensure(sizeof(Uint32) == 4, "Wrong size: Uint32");
    ensure(sizeof(Uint64) == 8, "Wrong size: Uint64");
    ensure(sizeof(Uint128) == 16, "Wrong size: Uint128");

    ensure(sizeof(Sint8) == 1, "Wrong size: Sint8");
    ensure(sizeof(Sint16) == 2, "Wrong size: Sint16");
    ensure(sizeof(Sint32) == 4, "Wrong size: Sint32");
    ensure(sizeof(Sint64) == 8, "Wrong size: Sint64");
    ensure(sizeof(Sint128) == 16, "Wrong size: Sint128");

    ensure(sizeof(Float32) == 4, "Wrong size: Float32");
    ensure(sizeof(Float64) == 8, "Wrong size: Float64");

    if(UINT64_MAX != 0xFFFFFFFFFFFFFFFFULL) {
        printf("Failed sanity check: UINT64_MAX\n");
        return 1;
    }

    if(N_MAX_CYCLES != 64) {
        // This is particularly annoying. See csim.pxd for explanation
        printf("Failed sanity check: N_MAX_CYCLES\n");
        return 8;
    }

    return 0;
}