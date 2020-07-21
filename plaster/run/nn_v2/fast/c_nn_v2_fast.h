#ifndef C_NN_V2_FAST_H
#define C_NN_V2_FAST_H


// See c_nn_v2_fast.c for docs.

#include "flann.h"

typedef __uint8_t Uint8;
typedef __uint32_t Uint32;
typedef __uint64_t Uint64;
typedef __uint128_t Uint128;
typedef __int64_t Sint64;
typedef __int8_t Sint8;
typedef float Float32;


typedef Uint64 Size;
typedef Uint32 Size32;
typedef Uint64 Index;
typedef Uint32 Index32;
typedef Uint8 DyeType;
typedef Float32 RadType;
typedef Float32 Score;
typedef Float32 WeightType;

#define N_MAX_NEIGHBORS (8)


typedef struct {
    Size count;
    Index dtr_i;
    Index pep_i;
} DyePepRec;


typedef struct {
    // The following are set by the .pyx file
    Size n_neighbors;
    Size n_cols;

    Size test_unit_radmat_n_rows;
    RadType *test_unit_radmat;

    Size train_dyemat_n_rows;
    RadType *train_dyemat;
    WeightType *train_dyetrack_weights;

    Index32 *output_pred_iz;
    Score *output_scores;

    // The following are internal to the .c file; freed by context_free()
    struct FLANNParameters flann_params;
    flann_index_t flann_index_id;
} Context;

void context_start(Context *ctx);
void context_free(Context *ctx);
void context_print(Context *ctx);
int test_flann();

#endif
