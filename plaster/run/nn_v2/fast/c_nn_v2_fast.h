#ifndef C_NN_V2_FAST_H
#define C_NN_V2_FAST_H


#include "c_common_old.h"
#include "flann.h"

typedef struct {
    // The following are set by the .pyx file
    Size n_neighbors;
    Size n_cols;
    Bool run_row_k_fit;
    Bool run_against_all_dyetracks;

    Tab test_radmat;  // RadType (Float32)
    Tab train_dyetrack_weights;  // WeightType (Float32)
    Tab train_dyemat;  // RadType (Float32)
    Tab train_dyepeps;  // (Index * 3)
    Tab train_dyt_i_to_dyepep_offset;  // (Index into train_dyepeps)
    Tab output_pred_pep_iz;  // Index32
    Tab output_pred_dyt_iz;  // Index32
    Tab output_scores;  // Score (Float32)
    Tab output_dists;  // Score (Float32)

    Float64 beta;
    Float64 sigma;
    Float64 zero_mu;
    Float64 zero_sigma;
    Float64 row_k_std;
    Size n_threads;
    Size n_rows;
    Index next_row_i;
    Size n_rows_per_block;
    pthread_mutex_t work_order_lock;
    pthread_mutex_t flann_index_lock;
    pthread_mutex_t pyfunction_lock;
    ProgressFn progress_fn;
    CheckKeyboardInterruptFn check_keyboard_interrupt_fn;

    // The following are internal to the .c file; freed by context_free()
    struct FLANNParameters flann_params;
    flann_index_t flann_index_id;
    int stop_requested;
} NNV2FastContext;


int context_start(NNV2FastContext *ctx);
void context_free(NNV2FastContext *ctx);
void context_print(NNV2FastContext *ctx);
int test_flann();

#endif
