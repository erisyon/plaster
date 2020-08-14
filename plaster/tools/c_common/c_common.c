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


//Table rle_index_init(Index *src, Size n_src, Index *dst, Size n_dst) {
//    // n_dst is +1 of the expected largest in src
//
//    Index minus_one = 0xFFFFFFFFFFFFFFFF;
//
//    Index last_i = minus_one;
//    Index dst_i = 0;
//    for(Index dst_i=0; dst_i<n_dst; dst_i++) {
//        dst[dst_i] = minus_one;
//    }
//    dst_i = 0;
//    for(Index src_i=0; src_i<n_src; src_i++) {
//        ensure(src[src_i] >= last_i || last_i == minus_one, "Illegal non sequential rle");
//        if (src[src_i] != last_i) {
//            dst[dst_i] = src_i;
//            dst_i++;
//            ensure(dst_i < n_dst-1);
//            last_i = src[src_i];
//        }
//    }
//    ensure(n_dst == dst_i)
//    dst[dst_i] = n_src;
//    return table_init_readonly(dst, sizeof(Index) * n_dst, sizeof(Index));
//}
//
//
//RLEBlock rle_index_get(Table *rle_table, Index pos) {
//    Index start = *table_get_row(rle_table, pos, Index);
//    Index stop = *table_get_row(rle_table, pos+1, Index);
//    ensure(start != minus_one && stop != minus_one, "rle accessing unknown block");
//    RLEBlock block;
//    block.i = start;
//    block.n = stop - start;
//    return block;
//}

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
