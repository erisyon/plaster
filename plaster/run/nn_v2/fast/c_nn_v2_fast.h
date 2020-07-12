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


typedef struct {
    Size dtr_i;
    Size pep_i;
    Size count;
} DyePepRec;


typedef struct {
    Size dt_i;
    Score score;
} CallRec;


typedef struct {
    Size n_cols;
    Size radmat_n_rows;
    RadType *radmat;

    Size train_dyetracks_n_rows;
    DyeType *train_dyetracks;
    Size train_dyepeps_n_rows;
    DyePepRec *train_dyepeps;

    struct FLANNParameters flann_params;
    flann_index_t flann_index_id;

    Float32 *dyetracks_as_floats;
} NNContext;


void context_work_orders_start(NNContext *ctx);
void context_free(NNContext *ctx);
int test_flann();


#endif
