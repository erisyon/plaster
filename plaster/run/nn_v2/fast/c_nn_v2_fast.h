#ifndef C_NN_V2_FAST_H
#define C_NN_V2_FAST_H


#include "c_common.h"
#include "flann.h"


typedef struct {
    // The following are set by the .pyx file
    Size n_neighbors;
    Size n_cols;

    Table test_unit_radmat;  // RadType (Float32)
    Table train_dyetrack_weights;  // WeightType (Float32)
    Table train_dyemat;  // RadType (Float32)
    Table train_dyepeps;  // (Index * 3)
    Table train_dye_i_to_dyepep_offset;  // (Index into train_dyepeps)
    Table output_pred_pep_iz;  // Index32
    Table output_pred_dye_iz;  // Index32
    Table output_scores;  // Score (Float32)

    Size n_threads;
    Size n_rows;
    Index next_row_i;
    Size n_rows_per_block;
    pthread_mutex_t work_order_lock;
    ProgressFn progress_fn;

    // The following are internal to the .c file; freed by context_free()
    struct FLANNParameters flann_params;
    flann_index_t flann_index_id;
} NNV2FastContext;


void context_start(NNV2FastContext *ctx);
void context_free(NNV2FastContext *ctx);
void context_print(NNV2FastContext *ctx);
int test_flann();

#endif
