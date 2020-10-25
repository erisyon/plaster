/*
Fast bounds-checking tables

Goals:
    - Bound checked table access
    - Easy passing around table with row sizes, etc
    - Easy debugging with __LINE__ and __FILE__ macros
    - NON DEBUG version that remove bound checking
    - growable

    DONE tab_by_size(void *base, Size n_bytes, Size n_bytes_per_row, int growable)
    DONE tab_by_n_rows(void *base, Size n_rows, Size n_bytes_per_row, int growable)
    DONE tab_row(Table *tab, Index row_i)
    DONE tab_var(typ, var, Table *tab, Index row_i)
    DONE tab_set(Table *tab, Index row_i, void *src)
    DONE tab_get(type, Table *tab, Index row_i)
    DONE tab_add(Table *tab, void *src, pthread_mutex_t *lock)
    DONE tab_validate(Table *tab)
    DONE tab_dump(Table *tab)
    tab_subset(void *base, Index start, Size n_rows)
    tab_tests()



#include "stdint.h"
#include "alloca.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdarg.h"
#include "memory.h"
#include "pthread.h"
#include "unistd.h"
#include "math.h"
#include "c_common_old.h"

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

#define TAB_NOT_GROWABLE (0)
#define TAB_GROWABLE (1)
#define TAB_NO_LOCK (void *)0


typedef struct {
    void *base;
    Uint64 n_bytes_per_row;
    Uint64 n_max_rows;
    Uint64 n_rows;
    int b_growable;
} Tab;


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



#define tab_row(tab, row_i) _tab_get(tab, row_i, __FILE__, __LINE__)
#define tab_var(typ, var, tab, row_i) typ *var = (typ *)_tab_get(tab, row_i, __FILE__, __LINE__)
#define tab_ptr(typ, tab, row_i) (typ *)_tab_get(tab, row_i, __FILE__, __LINE__)
#define tab_get(typ, tab, row_i) *(typ *)_tab_get(tab, row_i, __FILE__, __LINE__)
#define tab_set(tab, row_i, src) _tab_set(tab, row_i, src, __FILE__, __LINE__)
#define tab_add(tab, src, lock) _tab_add(tab, src, lock, __FILE__, __LINE__)
#define tab_validate(tab, ptr) _tab_validate(tab, ptr, __FILE__, __LINE__)


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


int main(int argc, char **argv) {
    tab_tests();
    return 0;
}
*/