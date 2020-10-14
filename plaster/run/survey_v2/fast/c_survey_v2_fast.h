#ifndef SURVEY_V2_H
#define SURVEY_V2_H

#include "c_common.h"
#include "flann.h"


typedef struct {
    Tab dyemat;
    Tab dyepeps;
    Tab pep_i_to_dyepep_row_i;
    Tab dyt_i_to_n_reads;
    Tab dyt_i_to_mlpep_i;
    Tab output_pep_i_to_isolation_metric;
    Tab output_pep_i_to_mic_pep_i;
    Index next_pep_i;
    Size n_threads;
    Size n_flann_cores;
    Size n_peps;
    Size n_neighbors;
    Size n_dyts;
    Size n_dyt_cols;
    Float32 distance_to_assign_an_isolated_pep;
    pthread_mutex_t work_order_lock;
    struct FLANNParameters flann_params;
    flann_index_t flann_index_id;
    ProgressFn progress_fn;
    Float32 p_func_k;
} SurveyV2FastContext;


void context_start(SurveyV2FastContext *ctx);


#endif
