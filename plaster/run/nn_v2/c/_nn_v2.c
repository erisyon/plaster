#include "flann.h"
#include "stdint.h"
#include "alloca.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdarg.h"
#include "memory.h"
#include "pthread.h"
#include "unistd.h"
#include "math.h"
#include "c_common.h"
#include "_nn_v2.h"


Float64 normalCDF(Float64 value) {
   return 0.5 * erfc(-value * M_SQRT1_2);
}


// TODO: Research best way choose standard_n_dz
#define standard_n_dz 0.01
Float64 p_value_from_z_score(Float64 z_score) {
    return normalCDF(z_score + standard_n_dz) - normalCDF(z_score - standard_n_dz);
}


Float64 p_from_gaussian(Float64 x, Float64 mu, Float64 sigma) {
    Float64 z_score = (x - mu) / sigma;
    return p_value_from_z_score(z_score);
}

/*
void score_weighted_gaussian_mixture(
    NNV2Context *ctx,
    int n_neighbors,
    Size n_cols,
    int *neighbor_dyt_iz,  // array((n_neighbors,), type=int): indices to dyetrack
    Tab *train_dyemat,  // arrays((n_dyetracks, n_cols), type=RadType): All dye weights
    RadType *radrow,  // arrays((n_cols,), type=RadType): radrow
    Tab *dyetrack_weights,  // arrays((n_dyetracks,), type=RadType): All dye weights
    ScoreType *output_scores  // array((n_neighbors,), type=float): returned scores for each neighbor
) {
    double weights[N_MAX_NEIGHBORS];
    double weighted_pdf[N_MAX_NEIGHBORS];
    double weighted_pdf_sum = 0.0;
    double std_per_dye = sqrt(0.1);

    for (int nn_i=0; nn_i<n_neighbors; nn_i++) {
        Index neighbor_i = neighbor_dyt_iz[nn_i];
        RadType *neighbor_target_dt = tab_ptr(RadType, train_dyemat, neighbor_i);
        DytWeightType neighbor_weight = tab_get(DytWeightType, ctx->_dyt_weights, neighbor_i);
        weights[nn_i] = (double)neighbor_weight;

        double vdist = (double)0.0;
        double det = 1.0;
        for (Index col_i=0; col_i<n_cols; col_i++) {
            double target_dt_for_col_i = (double)neighbor_target_dt[col_i];
            double delta = (double)radrow[col_i] - target_dt_for_col_i;

            double std_units = std_per_dye * (target_dt_for_col_i == 0.0 ? 0.5 : target_dt_for_col_i);
            double variance = std_units * std_units;
            ensure_only_in_debug(variance > 0, "Illegal zero variance");
            det *= variance;
            vdist += delta * delta / variance;
        }
        ensure_only_in_debug(det > 0, "Illegal zero det");
        double inv_sqrt_det = 1.0 / sqrt(det);
        double pdf = inv_sqrt_det * exp(-vdist / 2.0);
        double wpdf = (double)neighbor_weight * pdf;
        weighted_pdf[nn_i] = wpdf;
        weighted_pdf_sum += wpdf;

        if(ctx->_stop_requested) {
            return;
        }
    }

    for (int nn_i=0; nn_i<n_neighbors; nn_i++) {
        ScoreType penalty = (ScoreType)(1.0 - exp(-0.8 * weights[nn_i]));
        if(weighted_pdf_sum > 0.0) {
            ScoreType score_pre_penalty = (ScoreType)(weighted_pdf[nn_i] / weighted_pdf_sum);
            output_scores[nn_i] = penalty * score_pre_penalty;
        }
        else {
            output_scores[nn_i] = (ScoreType)0;
        }

        if(ctx->_stop_requested) {
            return;
        }
    }
}
*/

void score_k_fit_lognormal_mixture(
    NNV2Context *ctx,
    Size n_neighbors,
    Size n_cols,  // n_channels * n_cycles
    Tab *neighbor_dyt_iz,  // array((n_neighbors,), type=int): indices to dyetrack
    Tab *train_dyemat,  // arrays((n_dyetracks, n_cols), type=RadType): All dye weights
    RadType *radrow,  // arrays((n_cols,), type=RadType): radrow
    Tab *output_p_vals,  // array((n_neighbors,), type=float): returned scores for each neighbor
    Tab *output_pred_row_ks  // array((n_neighbors,), type=float): returned scores for each neighbor
) {
    // This is a log-normal model where the zero-counts (darks) are treated differently.
    // The non-zeros are log() and those mapping to zeros are not.
    // This requires that ctx contain beta, sigma for the lognormal and
    // zero_mu, zero_sigma for the normal of the zeros.

    Float64 beta = ctx->beta;
    Float64 sigma = ctx->sigma;
    Float64 zero_mu = ctx->zero_mu;
    Float64 zero_sigma = ctx->zero_sigma;
trace("here %d\n", __LINE__);

    for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
trace("here %d\n", __LINE__);
        Index neighbor_i = (Index)tab_get(int, neighbor_dyt_iz, nn_i);
        RadType *target_dt = tab_ptr(RadType, train_dyemat, neighbor_i);

        RadType adjusted_radrow[N_MAX_CHANNELS * N_MAX_CYCLES];
        RowKType pred_k = 1.0;
        if(ctx->run_row_k_fit) {
trace("here %d\n", __LINE__);
            // If fitting the k value for the row then solve for pred_k
            // and write the adjusted radrow into adjusted_radrow.
            // And then swap out the radrow pointer for this adjustment.
            RowKType sum_of_radrow_squares = 0.0;
            RowKType sum_of_radrow_beta_dyerow_products = 0.0;
            for(Index col_i=0; col_i<n_cols; col_i++) {
                sum_of_radrow_squares += radrow[col_i] * radrow[col_i];
                sum_of_radrow_beta_dyerow_products += radrow[col_i] * (target_dt[col_i] * beta);
            }
            if(sum_of_radrow_beta_dyerow_products > 0.0) {
trace("here %d %e\n", __LINE__, sum_of_radrow_beta_dyerow_products);
                pred_k = sum_of_radrow_squares / sum_of_radrow_beta_dyerow_products;
trace("here %d %f %f %f\n", __LINE__, sum_of_radrow_squares, sum_of_radrow_beta_dyerow_products, pred_k);
            }
            else {
                pred_k = 1.0;
trace("here %d %f\n", __LINE__, pred_k);
            }
            for(Index col_i=0; col_i<n_cols; col_i++) {
                adjusted_radrow[col_i] = radrow[col_i] / pred_k;
            }
            radrow = adjusted_radrow;
        }

trace("here %d\n", __LINE__);
        Float64 p_value = 1.0; // This is an accumulated product
        for(Index col_i=0; col_i<n_cols; col_i++) {
            Float64 rad, z_score;
            if(target_dt[col_i] > 0) {
                rad = log(max(1e-50, (Float64)radrow[col_i]));
                z_score = (rad - beta) / sigma;
            }
            else {
                rad = (Float64)radrow[col_i];
                z_score = (rad - zero_mu) / zero_sigma;
            }
            p_value *= p_value_from_z_score(z_score);
        }

trace("here %d %f %f\n", __LINE__, p_value, pred_k);
        tab_set(output_p_vals, nn_i, &p_value);
        tab_set(output_pred_row_ks, nn_i, &pred_k);
    }
trace("here %d\n", __LINE__);
}


#define check_and_return(expr, static_fail_string) if(!(expr)) return static_fail_string;

char *context_init(NNV2Context *ctx) {
    // Return 0 on success or pointer to a static string for an error

    float speedup = 0.0f;
    ctx->_flann_params = &DEFAULT_FLANN_PARAMETERS;
    ctx->_flann_index_id = flann_build_index_float(
        tab_ptr(RadType, &ctx->train_dyemat, 0),
        ctx->train_dyemat.n_rows,
        ctx->n_cols,
        &speedup,
        ctx->_flann_params
    );

    // COMPUTE weights and dyt_i_to_dyepep_offset lookup tables
    // Set last dye to a huge number so that it will be different on first
    // comparison with dyt_i below.
    Index last_dyt_i = 0xFFFFFFFFFFFFFFFF;
    Size last_count = 0;

    Size n_dyepep_rows = ctx->train_dyepeps.n_rows;
    Size n_dyts = ctx->train_dyemat.n_rows;

    //Size n_dyts = ctx->dyt_i_to_dyepep_offset.n_rows;

    ctx->_dyt_weights = tab_malloc_by_n_rows(n_dyts, sizeof(DytWeightType), TAB_NOT_GROWABLE);
    ctx->_dyt_i_to_dyepep_offset = tab_malloc_by_n_rows(n_dyts, sizeof(DytIndexType), TAB_NOT_GROWABLE);

    for(DytIndexType dyepep_i=0; dyepep_i<n_dyepep_rows; dyepep_i++) {
        DyePepType *dyepep = tab_ptr(DyePepType, &ctx->train_dyepeps, dyepep_i);

        Index dyt_i = dyepep[0];
        Index pep_i = dyepep[1];
        Index count = dyepep[2];
        ensure(0 <= dyt_i && dyt_i < n_dyts, "Bad dyt_i index %d", dyt_i);

        DytWeightType *w = tab_ptr(DytWeightType, &ctx->_dyt_weights, dyt_i);
        *w = *w + count;

        if(dyt_i != last_dyt_i) {
            check_and_return(dyt_i == last_dyt_i + 1 || last_dyt_i == 0xFFFFFFFFFFFFFFFF, "Non sequential dyt_i");
            check_and_return(0 <= dyt_i && dyt_i < n_dyts, "Illegal dyt_i when setting dyt_i_to_dyepep_offset");

            tab_set(&ctx->_dyt_i_to_dyepep_offset, dyt_i, &dyepep_i);
        }
        else {
            // Ensure that dyespeps is *reverse* sorted by count within each dyt
            // This allows choosing the most likely pep without a search
            check_and_return(count <= last_count, "train_dyepeps_view must be reverse sorted by count per dyt");
        }

        last_dyt_i = dyt_i;
        last_count = count;
    }

    return NULL;
}


void context_free(NNV2Context *ctx) {
    if(ctx->_flann_index_id) {
        flann_free_index_float(ctx->_flann_index_id, ctx->_flann_params);
        ctx->_flann_index_id = 0;
    }
    tab_free(&ctx->_dyt_weights);
    tab_free(&ctx->_dyt_i_to_dyepep_offset);
}


char *classify_radrows(
    NNV2Context *ctx,
    Index radrow_start_i,
    Size n_radrows
) {
    // Entrypoint for multi-core runners.
    // The context has everything that is needed to run the classifier
    // but this entrypoint allows a sliced batch to run.

    Size n_cols = ctx->n_cols;
    check_and_return(radrow_start_i <= 1024*16, "Too many rows (might overflow stack)");

    Size n_neighbors = 0;

    Tab *neighbor_dyt_iz = NULL;
// TODO
//    if(ctx->run_against_all_dyetracks) {
//        // In this mode there is no neighbor look up
//        // Each radrow is compared to every dyerow.
//        // This is implemented by setting the neighbor_dyt_iz to point to every dyt
//        Size n_dyts = ctx->train_dyemat.n_rows / n_cols;
//        n_neighbors = n_dyts;
//        Tab _neighbor_dyt_iz = tab_malloc_by_n_rows(n_neighbors, sizeof(int), TAB_NOT_GROWABLE);
//        neighbor_dyt_iz = &_neighbor_dyt_iz;
//        for(Index i=0; i<n_dyts; i++) {
//            tab_set(neighbor_dyt_iz, i, &i);
//        }
//    }
//    else {
        // In this mode we use FLANN to lookup candidate neighbors
        // TODO: This will need to sweep over pred_k
        n_neighbors = ctx->n_neighbors;
        Tab _neighbor_dyt_iz = tab_malloc_by_n_rows(n_radrows * n_neighbors, sizeof(int), TAB_NOT_GROWABLE);
        neighbor_dyt_iz = &_neighbor_dyt_iz;

        // FETCH a batch of neighbors from FLANN in one call.
// TODO
//        if(ctx->n_threads > 1) {
//            pthread_mutex_lock(&ctx->flann_index_lock);
//        }

        Tab neighbor_dists = tab_malloc_by_n_rows(n_radrows * n_neighbors, sizeof(float), TAB_NOT_GROWABLE);

        // flann_find_nearest_neighbors_index_float requires and int for indicies.
        // which is compiled to 32 bits and thus neighbor_dyt_iz is typed as int instead
        // of the usual bit-sized specific typing used throughout the rest of the code.
        flann_find_nearest_neighbors_index_float(
            ctx->_flann_index_id,
            tab_ptr(RadType, &ctx->radmat, radrow_start_i),
            n_radrows,
            tab_ptr(int, neighbor_dyt_iz, 0),
            tab_ptr(float, &neighbor_dists, 0),
            n_neighbors,
            ctx->_flann_params
        );
//        if(ctx->n_threads > 1) {
//            pthread_mutex_unlock(&ctx->flann_index_lock);
//        }
//    }

    Tab neighbor_p_vals = tab_malloc_by_n_rows(n_radrows * n_neighbors, sizeof(Float64), TAB_NOT_GROWABLE);
    Tab neighbor_pred_row_ks = tab_malloc_by_n_rows(n_radrows * n_neighbors, sizeof(Float64), TAB_NOT_GROWABLE);

    // Compare every radrow to the "neighbor" dytetracks.
    Index last_row = radrow_start_i + n_radrows;
    for (Index row_i=radrow_start_i; row_i<last_row; row_i++) {
        if(ctx->_stop_requested) {
            break;
        }

        RadType *radrow = tab_ptr(RadType, &ctx->radmat, row_i);

        Tab *row_neighbor_dyt_iz;
        // TODO
//        if(ctx->run_against_all_dyetracks) {
//            // In this mode the "neighbors" are actually ALL dyetracks.
//            row_neighbor_dyt_iz = neighbor_dyt_iz;
//        }
//        else {
            // In this mode the "neighbors" come form FLANN
            Tab _row_neighbor_dyt_iz = tab_subset(neighbor_dyt_iz, row_i * n_neighbors, n_neighbors);
            row_neighbor_dyt_iz = &_row_neighbor_dyt_iz;
//        }

        Tab row_neighbor_p_vals = tab_subset(&neighbor_p_vals, row_i * n_neighbors, n_neighbors);
        Tab neighbor_pred_row_ks = tab_subset(&neighbor_pred_row_ks, row_i * n_neighbors, n_neighbors);

trace("here %d\n", __LINE__);
        score_k_fit_lognormal_mixture(
            ctx,
            n_neighbors,
            n_cols,
            row_neighbor_dyt_iz,
            &ctx->train_dyemat,
            radrow,
            &row_neighbor_p_vals,
            &neighbor_pred_row_ks
        );
trace("here %d\n", __LINE__);

        // At this point there is a neighbor_p_vals for each neighbor dyt.
        // If ctx->run_row_k_fit is true then there is also a fit_k

        // Now each target score is computed by combining:
        //  * the p(target_dytrack) = how often this dyetrack is generated (the dyetrack weight)
        //  * the p(k) = the probability of the pred_k (assuming k fitting is enabled)
        //  * the rareness penalty (lowers score for assignment to very rare dyetracks)
        //  * The score is then normalized by the sum of all neighbors

        for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
            Float64 p_val = tab_get(Float64, &row_neighbor_p_vals, nn_i);

            Index dyt_i = (Index)tab_get(int, row_neighbor_dyt_iz, nn_i);
            DytWeightType target_weight = tab_get(DytWeightType, &ctx->_dyt_weights, dyt_i);

            Float64 penalty = (1.0 - exp(-0.8 * (Float64)target_weight));
            Float64 pred_row_k = tab_get(Float64, &neighbor_pred_row_ks, nn_i);
            Float64 p_row_k = p_from_gaussian(pred_row_k, 1.0, ctx->row_k_std);

            p_val *= penalty * target_weight * p_row_k;

            // UPDATE the p_vals with the score
            tab_set(&row_neighbor_p_vals, nn_i, &p_val);
        }
trace("here %d\n", __LINE__);

        // PICK target dyetrack with the highest score
        Float64 highest_score = (Float64)0;
        Float64 score_sum = (Float64)0;
        Index highest_score_i = 0;
        for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
            Float64 score = tab_get(Float64, &row_neighbor_p_vals, nn_i);
            if (score > highest_score) {
                highest_score = score;
                highest_score_i = nn_i;
            }
            score_sum += score;
        }

        Index most_likely_dyt_i = (Index)tab_get(int, row_neighbor_dyt_iz, highest_score_i);
        ScoreType dyt_score = highest_score / score_sum;

        // PICK peptide winner using Maximum Likelihood
        // the .pyx asserts that these are sorted by highest
        // count so we can just pick [0] from the correct dyepep
        Index dyepeps_offset = tab_get(Index, &ctx->_dyt_i_to_dyepep_offset, most_likely_dyt_i);
        Index *dyepeps_block = tab_ptr(Index, &ctx->train_dyepeps, dyepeps_offset);
        ensure_only_in_debug(most_likely_dyt_i == 0 || dyepeps_block[0] == most_likely_dyt_i, "dyepeps_block points to wrong block");
        Index most_likely_pep_i = dyepeps_block[1];

        DytWeightType weight = tab_get(DytWeightType, &ctx->_dyt_weights, most_likely_dyt_i);
        ScoreType pep_score = (ScoreType)dyepeps_block[2] / (ScoreType)weight;
        ScoreType score = dyt_score * pep_score;
trace("here %d\n", __LINE__);

        // Feels like this check could come out but it is safest as is.
        if(most_likely_dyt_i == 0) {
            score = (ScoreType)0.0;
        }

        // Set output
        Float64 output_fields[] = {
            most_likely_dyt_i,
            most_likely_pep_i,
            score
        };

        tab_set(&ctx->output, row_i, output_fields);
//        tab_set(&output_pred_dyt_iz, row_i, &most_likely_dyt_i);
//        tab_set(&output_pred_pep_iz, row_i, &most_likely_pep_i);
//        tab_set(&output_scores, row_i, &score);
    }
trace("here %d\n", __LINE__);

    tab_free(&neighbor_p_vals);
    tab_free(&neighbor_pred_row_ks);
    tab_free(neighbor_dyt_iz);
    tab_free(&neighbor_dists);

trace("here %d\n", __LINE__);

    return NULL;
}


/*
Index context_work_orders_pop(NNV2FastContext *ctx) {
    // NOTE: This return +1! So that 0 can be reserved.

    if(ctx->n_threads > 1) {
        pthread_mutex_lock(&ctx->work_order_lock);
    }

    Index i = ctx->next_row_i;
    ctx->next_row_i += ctx->n_rows_per_block;

    if(ctx->n_threads > 1) {
        pthread_mutex_unlock(&ctx->work_order_lock);
    }

    if(i < ctx->n_rows) {
        return i + 1;
    }
    return 0;
}

void progress_thread_safe(NNV2FastContext* ctx, int complete, int total, int retry) {
    if(ctx->n_threads > 1) {
        pthread_mutex_lock(&ctx->pyfunction_lock);
    }

    ctx->progress_fn(complete, total, retry);

    if(ctx->n_threads > 1) {
        pthread_mutex_unlock(&ctx->pyfunction_lock);
    }
}

typedef struct {
    pthread_t id;
    int complete;
    NNV2FastContext *ctx;
} ThreadContext;


void *context_work_orders_worker(void *_tctx) {
    // The worker thread. Pops off which pep to work on next
    // continues until there are no more work orders.
    ThreadContext* tctx= (ThreadContext *)_tctx;
    NNV2FastContext *ctx = tctx->ctx;
    while(!ctx->stop_requested) {
        Index row_i_plus_1 = context_work_orders_pop(ctx);
        if(row_i_plus_1 == 0) {
            break;
        }
        Index row_i = row_i_plus_1 - 1;
        context_classify_radrows(
            ctx,
            tab_subset(&ctx->test_radmat, row_i, ctx->n_rows_per_block),
            tab_subset(&ctx->output_pred_pep_iz, row_i, ctx->n_rows_per_block),
            tab_subset(&ctx->output_pred_dyt_iz, row_i, ctx->n_rows_per_block),
            tab_subset(&ctx->output_scores, row_i, ctx->n_rows_per_block),
            tab_subset(&ctx->output_dists, row_i, ctx->n_rows_per_block)
        );
        progress_thread_safe(ctx, row_i, ctx->n_rows, 0);
    }
    progress_thread_safe(ctx, ctx->n_rows, ctx->n_rows, 0);
    tctx->complete = 1;
    return (void *)0;
}


int context_start(NNV2FastContext *ctx) {
    ensure(sanity_check() == 0, "Sanity checks failed");
    ensure(
        ctx->n_neighbors <= ctx->train_dyemat.n_rows,
        "FLANN does not support requesting more neighbors than there are data points"
    );
    ensure(
        ctx->n_neighbors <= N_MAX_NEIGHBORS,
        "Too many neighbors requested"
    );

    // context_print(ctx);

    // CLEAR internally controlled elements
    ctx->flann_params = DEFAULT_FLANN_PARAMETERS;
    ctx->flann_index_id = 0;
    ctx->flann_params.cores = 1;
    ctx->stop_requested = 0;

    // CREATE the ANN index
    float speedup = 0.0f;
    ctx->flann_index_id = flann_build_index_float(
        tab_ptr(RadType, &ctx->train_dyemat, 0),
        ctx->train_dyemat.n_rows,
        ctx->n_cols,
        &speedup,
        &ctx->flann_params
    );

    ThreadContext tctxs[256];
    ensure(0 < ctx->n_threads && ctx->n_threads < 256, "Invalid n_threads");

    if(ctx->n_threads > 1) {
        int ret = pthread_mutex_init(&ctx->work_order_lock, NULL);
        ensure(ret == 0, "pthread lock create failed");

        ret = pthread_mutex_init(&ctx->flann_index_lock, NULL);
        ensure(ret == 0, "pthread lock create failed");

        ret = pthread_mutex_init(&ctx->pyfunction_lock, NULL);
        ensure(ret == 0, "pthread lock create failed");
    }

    for(Index i=0; i<ctx->n_threads; i++) {
        tctxs[i].ctx = ctx;
        tctxs[i].complete = 0;
        int ret = pthread_create(&tctxs[i].id, NULL, context_work_orders_worker, &tctxs[i]);
        ensure(ret == 0, "Thread not created.");
    }

    int interrupted = 0;
    while(1) {
        int complete = 1;
        for(Index i=0; i<ctx->n_threads; i++) {
            if(!tctxs[i].complete) {
                complete = 0;
                break;
            }
        }

        if(ctx->n_threads > 1) {
            pthread_mutex_lock(&ctx->pyfunction_lock);
        }
        int got_interrupt = ctx->check_keyboard_interrupt_fn();
        if(ctx->n_threads > 1) {
            pthread_mutex_unlock(&ctx->pyfunction_lock);
        }

        if(got_interrupt) {
            printf("Ctrl-C received, please wait a few seconds until all threads complete\n");
            ctx->stop_requested = 1;
            interrupted = 1;
            break;
        }

        if(complete) {
            break;
        }
        // Calling ctx->check_keyboard_interrupt_fn too often results in a segfault
        usleep(50000);
    }

    for(Index i=0; i<ctx->n_threads; i++) {
        pthread_join(tctxs[i].id, NULL);
    }

    return interrupted;
}


void context_free(NNV2FastContext *ctx) {
    if(ctx->flann_index_id) {
        flann_free_index_float(ctx->flann_index_id, &ctx->flann_params);
        ctx->flann_index_id = 0;
    }
}


void context_print(NNV2FastContext *ctx) {
    printf("n_neighbors=%ld\n", ctx->n_neighbors);
    printf("n_cols=%ld\n", ctx->n_cols);
    printf("run_against_all_dyetracks=%ld\n", ctx->run_against_all_dyetracks);
    printf("train_dyemat.n_rows=%ld\n", ctx->train_dyemat.n_rows);
    printf("test_radmat.n_rows=%ld\n", ctx->test_radmat.n_rows);
    for(Index row_i=0; row_i<ctx->test_radmat.n_rows; row_i++) {
        RadType *radrow = tab_ptr(RadType, &ctx->test_radmat, row_i);
        for(Index c=0; c<ctx->n_cols; c++) {
            printf("%2.1f ", radrow[c]);
        }
        printf("\n");
    }
}
*/