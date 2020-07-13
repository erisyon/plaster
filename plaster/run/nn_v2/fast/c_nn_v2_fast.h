#ifndef C_NN_V2_FAST_H
#define C_NN_V2_FAST_H


// See c_nn_v2_fast.c for docs.

#include "flann.h"

typedef __uint8_t Uint8;
typedef __uint64_t Uint64;
typedef __uint128_t Uint128;
typedef __int64_t Sint64;
typedef __int8_t Sint8;
typedef float Float32;


typedef Uint64 Size;
typedef Uint64 Index;
typedef Uint8 DyeType;
typedef Float32 RadType;
typedef Float32 Score;

#define N_MAX_NEIGHBORS (8)


typedef struct {
    Size dtr_i;
    Size pep_i;
    Size count;
} DyePepRec;


typedef struct {
    Index dt_i;
    Score score;
} CallRec;


typedef struct {
    // Set by the .pyx file
    Size n_cols;
    Size radmat_n_rows;
    RadType *radmat;

    Size train_dyetracks_n_rows;
    DyeType *train_dyetracks;
    Size train_dyepeps_n_rows;
    DyePepRec *train_dyepeps;

    CallRec *output_callrecs;

    // Internal to the .c file; freed by context_free()
    struct FLANNParameters flann_params;
    flann_index_t flann_index_id;
    RadType *dyetracks_as_radtype;
    RadType *dyetrack_weights;
} Context;

void context_start(Context *ctx);
void context_free(Context *ctx);
int test_flann();


#endif
`
