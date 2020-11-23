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
    if(isnan(z_score)) {
        return 0.0;
    }
    ensure(z_score >= 0.0, "z_score must be non-negative %f", z_score);
    return normalCDF(z_score + standard_n_dz) - normalCDF(z_score - standard_n_dz);
}


Float64 p_from_gaussian(Float64 x, Float64 mu, Float64 sigma) {
    if(sigma == 0.0 && x == mu) {
        // Special case for sigma == 0.0 meaning that a paramter is not changing
        return 1.0;
    }
    Float64 z_score = fabs(x - mu) / sigma;
    return p_value_from_z_score(z_score);
}


char *score_k_fit_lognormal_mixture(
    NNV2Context *ctx,
    Size n_neighbors,
    Size n_cols,  // n_channels * n_cycles
    Tab *neighbor_dyt_iz,  // array((n_neighbors,), type=int): indices to dyetrack
    Tab *train_fdyemat,  // arrays((n_dyetracks, n_cols), type=RadType): All dye weights
    RadType *radrow,  // arrays((n_cols,), type=RadType): radrow
    Tab *output_p_vals,  // array((n_neighbors,), type=float): returned scores for each neighbor
    Tab *output_pred_row_ks,  // array((n_neighbors,), type=float): returned scores for each neighbor
    Tab *output_sum_log_z_scores  // array((n_neighbors,), type=float):
) {
    // This is a log-normal model where the zero-counts (darks) are treated differently.
    // The non-zeros are log() and those mapping to zeros are not.
    // This requires that ctx contain beta, sigma for the lognormal and
    // zero_beta, zero_sigma for the normal of the zeros.

    Index neighbor_dyt_0 = (Index)tab_get(int, neighbor_dyt_iz, 0);
    if(neighbor_dyt_0 == 0) {
        for(Index col_i=0; col_i<n_cols; col_i++) {
            RadType val = tab_col(RadType, train_fdyemat, 0, col_i);
            check_and_return(val == 0, "dyt_i must be all zero");
        }
    }

    Float64 beta = ctx->beta;
    Float64 sigma = ctx->sigma;
    Float64 zero_beta = ctx->zero_beta;
    Float64 zero_sigma = ctx->zero_sigma;

    for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
        Index neighbor_dyt_i = (Index)tab_get(int, neighbor_dyt_iz, nn_i);
        RadType *target_dt = tab_ptr(RadType, train_fdyemat, neighbor_dyt_i);

        if(neighbor_dyt_i == 0) {
            // This is a check against the nul-dyt which has must
            // be treated specially to avoid div 0 and similar issues.
            Float64 zero = 0.0;
            tab_set(output_p_vals, nn_i, &zero);
            tab_set(output_sum_log_z_scores, nn_i, &zero);
            tab_set(output_pred_row_ks, nn_i, &zero);
            continue;
        }

        RadType adjusted_radrow[N_MAX_CHANNELS * N_MAX_CYCLES];
        RowKType pred_k = 1.0;
        if(ctx->run_row_k_fit) {
            // If fitting the k value for the row then solve for pred_k
            // and write the adjusted radrow into adjusted_radrow.
            // And then swap out the radrow pointer for this adjustment.
            RowKType sum_of_radrow_squares = 0.0;
            RowKType sum_of_radrow_beta_dyerow_products = 0.0;
            for(Index col_i=0; col_i<n_cols; col_i++) {
                sum_of_radrow_squares += radrow[col_i] * radrow[col_i];
                sum_of_radrow_beta_dyerow_products += radrow[col_i] * target_dt[col_i];
            }
            if(sum_of_radrow_beta_dyerow_products > 0.0) {
                pred_k = sum_of_radrow_squares / sum_of_radrow_beta_dyerow_products;
            }
            else {
                // In this case we make pred_k be an arbitrarily large
                // value so that the division below will succeed but
                // it will push the adjusted_radrow to effectively zero
                // and then the p_row_k will also be ~zero.
                pred_k = 1e1;
            }
        }

        for(Index col_i=0; col_i<n_cols; col_i++) {
            adjusted_radrow[col_i] = radrow[col_i] / pred_k;
        }

        Float64 log_beta = log(beta);
        Float64 p_value = 1.0; // This is an accumulated product
        Float64 sum_log_z_score = 0.0;
        Float64 sum_log_p = 0.0;
        for(Index col_i=0; col_i<n_cols; col_i++) {
            Float64 rad, z_score;
            if(target_dt[col_i] > 0) {
                rad = log(max(1e-50, (Float64)adjusted_radrow[col_i]));
                z_score = (rad - log(target_dt[col_i])) / sigma;
            }
            else {
                rad = (Float64)adjusted_radrow[col_i];
                z_score = (rad - zero_beta) / zero_sigma;
            }
            z_score = fabs(z_score);
            sum_log_z_score += log(z_score);
            Float64 p = p_value_from_z_score(z_score); 
            p_value *= p;
            sum_log_p += log(p);
        }

        // EXPERIMENT: use the mean
        // This does indeed make the k-dist 1-centered but now
        // I suspect that k is too-important. Will need to sweep this
        p_value = exp(sum_log_p / (Float64)n_cols);

        tab_set(output_p_vals, nn_i, &p_value);
        tab_set(output_sum_log_z_scores, nn_i, &sum_log_z_score);
        tab_set(output_pred_row_ks, nn_i, &pred_k);
    }

    return NULL;
}


void validate_radmat_table(NNV2Context *ctx, char *msg) {
    Size n_rows = ctx->radmat.n_rows;
    for(int r=0; r<n_rows; r++) {
        RadType *rr = tab_ptr(RadType, &ctx->radmat, r);
        for(int c=0; c<ctx->n_cols; c++) {
            if(!(-10000.0 < rr[c] && rr[c] < 1e6)) {
                fprintf(stderr, "r=%d c=%d  val=%f\n", r, c, rr[c]);
                ensure(0, "bad radmat %s", msg);
                exit(1);
            }
        }
    }
}

static int flann_index_lock_initialized = 0;
static pthread_mutex_t flann_index_lock;

char *context_init(NNV2Context *ctx) {
    // validate_radmat_table(ctx, "context init");

    float speedup = 0.0f;
    ctx->_flann_params = &DEFAULT_FLANN_PARAMETERS;

    check_and_return(
        ctx->n_neighbors <= ctx->train_fdyemat.n_rows,
        "Requesting more neighbors than training rows"
    );

    check_and_return(
        ctx->row_k_sigma > 0.0 || ! ctx->use_row_k_p_val,
        "row_k_sigma was not set to a reasonable value"
    );

    // Scale fdyemat up by beta
    Float64 beta = ctx->beta;
    Size n_elems = ctx->train_fdyemat.n_rows * ctx->train_fdyemat.n_cols;
    RadType *elem = tab_ptr(RadType, &ctx->train_fdyemat, 0);
    for(Index i=0; i<n_elems; i++) {
        *elem++ = *elem * beta;
    }

    if (flann_index_lock_initialized == 0) {
        int ret = pthread_mutex_init(&flann_index_lock, NULL);
        check_and_return(
            ret == 0,
            "flann_index_lock_initialized failed to initialize"
        );
        flann_index_lock_initialized = 1;
    }
    ctx->_flann_index_lock = (struct pthread_mutex_t *)&flann_index_lock;

    if(ctx->_flann_index_lock != NULL) {
        pthread_mutex_lock((pthread_mutex_t *)ctx->_flann_index_lock);
    }

    ctx->_flann_index_id = flann_build_index_float(
        tab_ptr(RadType, &ctx->train_fdyemat, 0),
        ctx->train_fdyemat.n_rows,
        ctx->n_cols,
        &speedup,
        ctx->_flann_params
    );

    if(ctx->_flann_index_lock != NULL) {
        pthread_mutex_unlock((pthread_mutex_t *)ctx->_flann_index_lock);
    }

    // COMPUTE weights and dyt_i_to_dyepep_offset lookup tables
    // Set last dye to a huge number so that it will be different on first
    // comparison with dyt_i below.
    Index last_dyt_i = 0xFFFFFFFFFFFFFFFF;
    Size last_count = 0;

    Size n_dyepep_rows = ctx->train_dyepeps.n_rows;
    Size n_dyts = ctx->train_fdyemat.n_rows;

    //Size n_dyts = ctx->dyt_i_to_dyepep_offset.n_rows;

    // _dyt_weights is the sum of each dytetrack's counts. That is, how many
    // total times was this dyetrack generated by all peptides.
    ctx->_dyt_weights = tab_malloc_by_n_rows(n_dyts, sizeof(DytWeightType), TAB_NOT_GROWABLE);

    // _dyt_i_to_dyepep_offset is a lookup table that indexes into the dyepep table
    // by dyt_i (dyepeps table is sorted by dyt_i).
    // This LUT prevents having to search the dyepeps for the start of the a dyt_i block.
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

    pthread_mutex_destroy((pthread_mutex_t *)ctx->_flann_index_lock);
    flann_index_lock_initialized = 0;
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
    check_and_return(n_radrows <= 1024*16, "Too many rows (might overflow stack)");

    Size n_neighbors = 0;

    Tab *neighbor_dyt_iz = NULL;
    Tab *neighbor_dists = NULL;

    if( ! ctx->run_against_all_dyetracks) {
        // In this mode we use FLANN to lookup candidate neighbors
        // TODO: This will need to sweep over pred_k
        n_neighbors = ctx->n_neighbors;
        Tab _neighbor_dyt_iz = tab_malloc_by_n_rows(n_radrows * n_neighbors, sizeof(int), TAB_NOT_GROWABLE);
        neighbor_dyt_iz = &_neighbor_dyt_iz;

        if(ctx->_flann_index_lock != NULL) {
            pthread_mutex_lock((pthread_mutex_t *)ctx->_flann_index_lock);
        }

        // FETCH a batch of neighbors from FLANN in one call.
        Tab _neighbor_dists = tab_malloc_by_n_rows(n_radrows * n_neighbors, sizeof(float), TAB_NOT_GROWABLE);
        neighbor_dists = &_neighbor_dists;
        // flann_find_nearest_neighbors_index_float requires and int for indicies.
        // which is compiled to 32 bits and thus neighbor_dyt_iz is typed as int instead
        // of the usual bit-sized specific typing used throughout the rest of the code.

        int flann_ret = flann_find_nearest_neighbors_index_float(
            ctx->_flann_index_id,
            tab_ptr(RadType, &ctx->radmat, radrow_start_i),
            n_radrows,
            tab_ptr(int, neighbor_dyt_iz, 0),
            tab_ptr(float, neighbor_dists, 0),
            n_neighbors,
            ctx->_flann_params
        );
        ensure(flann_ret == 0, "FLANN returned an error. %d", flann_ret);

        if(ctx->_flann_index_lock != NULL) {
            pthread_mutex_unlock((pthread_mutex_t *)ctx->_flann_index_lock);
        }
    }
    else {
        // In this mode there is no neighbor look-up; rather ALL dytracks are treated
        // as if they were neighbors that were returned by FLANN thus making it so that
        // every radrow is compared to every dyerow.
        //
        // The complication comes in neighbor_dyt_iz
        // In FLANN mode, the neighbor_dyt_iz would be filled in by the flann_find_nearest_neighbors_index_float
        // call such that each radrow would have n_neighbors entries in this array
        // and that is used for comparison below.
        // But in this mode, we need an identity lookup table -- that is
        // a fully dense neighbor lookup table.

        Size n_dyts = ctx->train_fdyemat.n_rows;
        n_neighbors = n_dyts; // Fools following code in to using EVERY dyt as a neighbor

        // The below code that usually deals with WITH has to do a lookup to
        // convert from the return values that FLANN
        Tab _neighbor_dyt_iz = tab_malloc_by_n_rows(n_neighbors, sizeof(int), TAB_NOT_GROWABLE);
        neighbor_dyt_iz = &_neighbor_dyt_iz;
        for(Index i=0; i<n_dyts; i++) {
            tab_set(neighbor_dyt_iz, i, &i);
        }
    }

    // TODO: Consolidate into one table with multi-columns
    Tab neighbor_p_vals = tab_malloc_by_n_rows(n_radrows * n_neighbors, sizeof(Float64), TAB_NOT_GROWABLE);
    Tab neighbor_pred_row_ks = tab_malloc_by_n_rows(n_radrows * n_neighbors, sizeof(Float64), TAB_NOT_GROWABLE);
    Tab neighbor_output_sum_log_z_scores = tab_malloc_by_n_rows(n_radrows * n_neighbors, sizeof(Float64), TAB_NOT_GROWABLE);
    Tab neighbor_scores = tab_malloc_by_n_rows(n_radrows * n_neighbors, sizeof(Float64), TAB_NOT_GROWABLE);

    // Compare every radrow to the "neighbor" dytetracks.
    for (Index row_i=0; row_i<n_radrows; row_i++) {
        if(ctx->_stop_requested) {
            break;
        }

        // context_row_i is the index into the context's radmat
        Index context_row_i = row_i + radrow_start_i;

        RadType *radrow = tab_ptr(RadType, &ctx->radmat, context_row_i);
        Tab *row_neighbor_dyt_iz;
        if( ! ctx->run_against_all_dyetracks) {
            // In this mode the "neighbors" come form FLANN
            Tab _row_neighbor_dyt_iz = tab_subset(neighbor_dyt_iz, row_i * n_neighbors, n_neighbors);
            row_neighbor_dyt_iz = &_row_neighbor_dyt_iz;
        }
        else {
            // In this mode the "neighbors" are actually ALL dyetracks.
            row_neighbor_dyt_iz = neighbor_dyt_iz;
        }

        Tab row_neighbor_p_vals = tab_subset(&neighbor_p_vals, row_i * n_neighbors, n_neighbors);
        Tab row_neighbor_pred_row_ks = tab_subset(&neighbor_pred_row_ks, row_i * n_neighbors, n_neighbors);
        Tab row_neighbor_output_sum_log_z_scores = tab_subset(&neighbor_output_sum_log_z_scores, row_i * n_neighbors, n_neighbors);
        Tab row_neighbor_scores = tab_subset(&neighbor_scores, row_i * n_neighbors, n_neighbors);

        char *fail = score_k_fit_lognormal_mixture(
            ctx,
            n_neighbors,
            n_cols,
            row_neighbor_dyt_iz,
            &ctx->train_fdyemat,
            radrow,
            &row_neighbor_p_vals,
            &row_neighbor_pred_row_ks,
            &row_neighbor_output_sum_log_z_scores
        );
        check_and_return(!fail, fail);

        // At this point there is a neighbor_p_vals for each neighbor dyt.
        // If ctx->run_row_k_fit is true then there is also a fit_k

        // Now each target score is computed by combining:
        //  * the p(target_dytrack) = how often this dyetrack is generated (the dyetrack weight)
        //  * the p(k) = the probability of the pred_k (assuming k fitting is enabled)
        //  * the rareness penalty (lowers score for assignment to very rare dyetracks)
        //  * The score is then normalized by the sum of all neighbors

        Uint64 sum_target_weights = 0;
        for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
            Index dyt_i = (Index)tab_get(int, row_neighbor_dyt_iz, nn_i);
            Uint64 dyt_weight = (Uint64)tab_get(DytWeightType, &ctx->_dyt_weights, dyt_i);
            sum_target_weights += dyt_weight;
        }

        for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
            Float64 p_val = tab_get(Float64, &row_neighbor_p_vals, nn_i);

            Index dyt_i = (Index)tab_get(int, row_neighbor_dyt_iz, nn_i);
            DytWeightType target_weight = tab_get(DytWeightType, &ctx->_dyt_weights, dyt_i);
            Float64 penalty = (1.0 - exp(-0.8 * (Float64)target_weight));
            Float64 pred_row_k = tab_get(Float64, &row_neighbor_pred_row_ks, nn_i);
            Float64 p_row_k = p_from_gaussian(pred_row_k, ctx->row_k_beta, ctx->row_k_sigma);
            Float64 sum_log_z_score = tab_get(Float64, &row_neighbor_output_sum_log_z_scores, nn_i);

            if( ! ctx->use_row_k_p_val) {
                p_row_k = 1.0;
            }

            Float64 normalized_target_weight = 0.0;
            if(sum_target_weights > 0) {
                // For example, the nul-dytrack can have no true weight so
                // we set the normalized_target_weight tp zero in this case
                // to avoid the divide by 0.
                normalized_target_weight = (Float64)target_weight / (Float64)sum_target_weights;
            }

            if(ctx->run_against_all_dyetracks) {
                // In all mode, the weight and the penalty do not count;
                target_weight = 1.0;
                normalized_target_weight = 1.0;
            }

            // I think it would make sense to convert all these scores to logs
            // but for now I'm going to experiment with using power
            // Experiment: reduce importance of p_row_k
            Float64 composite_score = p_val * penalty * normalized_target_weight * pow(p_row_k, ctx->row_k_score_factor);

            // SET the row score
            tab_set(&row_neighbor_scores, nn_i, &composite_score);

            if(ctx->run_against_all_dyetracks) {
                // In this mode there are extra outputs to return
                tab_set_col(&ctx->against_all_dyetracks_output, context_row_i, nn_i, &p_val);
                tab_set_col(&ctx->against_all_dyetracks_output, context_row_i, 1*n_neighbors + nn_i, &pred_row_k);
            }
        }

        // PICK target dyetrack with the highest score
        Float64 highest_score = (Float64)0;
        Float64 score_sum = (Float64)0;
        Index highest_score_i = 0;
        for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
            Float64 score = tab_get(Float64, &row_neighbor_scores, nn_i);
            if (score > highest_score) {
                highest_score = score;
                highest_score_i = nn_i;
            }
            score_sum += score;
        }

        Index most_likely_dyt_i = (Index)tab_get(int, row_neighbor_dyt_iz, highest_score_i);
        Float64 most_likely_pred_k = tab_get(Float64, &row_neighbor_pred_row_ks, highest_score_i);
        Float64 logp_dyt = log(tab_get(Float64, &row_neighbor_p_vals, highest_score_i));
        Float64 logp_k = log(p_from_gaussian(most_likely_pred_k, ctx->row_k_beta, ctx->row_k_sigma));

        ScoreType dyt_score = highest_score / score_sum;

        Float64 most_likely_sum_log_z_score = tab_get(Float64, &row_neighbor_output_sum_log_z_scores, highest_score_i);

        // PICK peptide winner using Maximum Likelihood
        // The dyepeps are sorted so that the most likely peptide is first
        // so we can just pick the [0] from the dyepeps block to find the most likely pep_i
        Index dyepeps_offset = tab_get(Index, &ctx->_dyt_i_to_dyepep_offset, most_likely_dyt_i);
        Index *dyepeps_block = tab_ptr(Index, &ctx->train_dyepeps, dyepeps_offset);
        ensure_only_in_debug(most_likely_dyt_i == 0 || dyepeps_block[0] == most_likely_dyt_i, "dyepeps_block points to wrong block");
        DyePepType most_likely_pep_i = dyepeps_block[1];

        // "count_of_most_likely_pep_i" is the number of times that the most-likely-peptide
        // generated the most-likely-dyt.
        DyePepType count_of_most_likely_pep_i = dyepeps_block[2];

        // "most_likely_dyt_weight" was computed above: the total number of counts to a given dyt.
        DytWeightType most_likely_dyt_weight = tab_get(DytWeightType, &ctx->_dyt_weights, most_likely_dyt_i);

        // "pep_score" is the fraction of most_likely_dyt_i's calls that
        // are assigned to the most-likely-peptide which is the
        // count_of_most_likely_pep_i / most_likely_dyt_weight.
        ScoreType pep_score = (ScoreType)dyepeps_block[2] / (ScoreType)most_likely_dyt_weight;

        // "output_score" is dyt_score * pep_score ...
        ScoreType output_score = dyt_score * pep_score;
        if(most_likely_dyt_i == 0) {
            // Feels like this check could come out but it is safest as is.
            output_score = (ScoreType)0.0;
        }

        // Set output
        Float64 output_fields[] = {
            most_likely_dyt_i,
            most_likely_pep_i,
            dyt_score,
            output_score,
            most_likely_pred_k,
            logp_dyt,
            logp_k,
        };

        tab_set(&ctx->output, context_row_i, output_fields);
    }

    tab_free(&neighbor_p_vals);
    tab_free(&neighbor_pred_row_ks);
    tab_free(&neighbor_output_sum_log_z_scores);
    tab_free(neighbor_dyt_iz);
    tab_free(&neighbor_scores);
    if (neighbor_dists) {
        tab_free(neighbor_dists);
    }

    return NULL;
}
