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

Table table_init(Uint8 *base, Size n_bytes, Size n_bytes_per_row) {
    // Wrap an externally allocated memory buffer ptr as a Table.
    ensure(n_bytes_per_row % 8 == 0, "Mis-aligned table row size");
    ensure((Uint64)base % 8 == 0, "Mis-aligned table");
    Table table;
    table.rows = base;
    table.n_rows = 0;
    table.n_bytes_per_row = n_bytes_per_row;
    table.n_max_rows = n_bytes / n_bytes_per_row;
    memset(table.rows, 0, n_bytes);
    return table;
}


void *_table_get_row(Table *table, Index row) {
    // Fetch a row from a table with bounds checking if activated
    ensure_only_in_debug(0 <= row && row < table->n_rows, "table get outside bounds");
    return (void *)(table->rows + table->n_bytes_per_row * row);
}


Index table_add(Table *table, void *src, pthread_mutex_t *lock) {
    // Add a row to the table and halt on overflow.
    // Optionally copies src into place if it isn't NULL.
    // Returns the row_i where the data was written (or will be written)
    // This is a potential race condition; a mutex may be warranted.  TODO: Test
    if(lock) pthread_mutex_lock(lock);
    Index row_i = table->n_rows;
    table->n_rows ++;
    if(lock) pthread_mutex_unlock(lock);
    ensure(0 <= row_i && row_i < table->n_max_rows, "Table overflow");
    if(src != 0) {
        memcpy(table->rows + table->n_bytes_per_row * row_i, src, table->n_bytes_per_row);
    }
    return row_i;
}


void table_validate(Table *table, void *ptr, char *msg) {
    // Check that a ptr is valid on a table
    // Use table_validate_only_in_debug (see below)
    Sint64 byte_offset = ((Uint8 *)ptr - table->rows);
    ensure(byte_offset % table->n_bytes_per_row == 0, msg);
    ensure(0 <= byte_offset && (Size)byte_offset < table->n_bytes_per_row * table->n_max_rows, msg);
}


