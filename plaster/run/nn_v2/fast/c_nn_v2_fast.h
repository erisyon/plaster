#ifndef C_NN_V2_FAST_H
#define C_NN_V2_FAST_H


#include "c_common.h"
#include "flann.h"


typedef struct {
    // The following are set by the .pyx file
    Size n_neighbors;
    Size n_cols;

    Table test_unit_radmat;
    Table train_dyetrack_weights;
    Table train_dyemat;
    Table output_pred_iz;
    Table output_scores;

    // The following are internal to the .c file; freed by context_free()
    struct FLANNParameters flann_params;
    flann_index_t flann_index_id;
} Context;


void context_start(Context *ctx);
void context_free(Context *ctx);
void context_print(Context *ctx);
int test_flann();

#endif
