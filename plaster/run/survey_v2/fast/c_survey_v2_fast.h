#ifndef SURVEY_V2_H
#define SURVEY_V2_H

#include "c_common.h"
#include "flann.h"


typedef Float32 IsolationType;


typedef struct {
    Table dyemat;
    Table dyepeps;
    Table pep_i_to_dyepep_row_i;
    Table dyt_i_to_mlpep_i;
    Index next_pep_i;
    Size n_threads;
    Size n_peps;
    Size n_neighbors;
    Size n_dyts;
    Size n_dyt_cols;
    Table output_pep_i_to_isolation_metric;
    Float32 distance_to_assign_an_isolated_pep;
    pthread_mutex_t work_order_lock;
    struct FLANNParameters flann_params;
    flann_index_t flann_index_id;
    ProgressFn progress_fn;
} Context;


void context_start(Context *ctx);


#endif
