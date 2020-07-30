#include "stdint.h"
#include "alloca.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdarg.h"
#include "memory.h"
#include "pthread.h"
#include "unistd.h"
#include "math.h"
#include "c_nn_v2_fast.h"


int sanity_check() {
    if(sizeof(Float32) != 4) {
        printf("Failed sanity check: sizeof Float32\n");
        return 1;
    }

    if(sizeof(Score) != 4) {
        printf("Failed sanity check: sizeof Score\n");
        return 1;
    }

    if(sizeof(RadType) != 4) {
        printf("Failed sanity check: sizeof RadType\n");
        return 1;
    }

    return 0;
}


float dist_inv_square(RadType *radrow, RadType *dyerow, Size n_cols) {
    RadType *rad = radrow;
    RadType *dye = dyerow;
    RadType sq_dist = (RadType)0;
    for (Index i=0; i<n_cols; i++) {
        RadType delta = *rad - *dye;
        sq_dist += delta * delta;
    }
    return (RadType)1 / sq_dist;
}


void score_weighted_inv_square(
    int n_neighbors,
    int *neighbor_dye_iz,  // array((n_neighbors,), type=int): indices to dyetrack
    float *neighbor_dists,  // array((n_neighbors,), type=float): distances computed by FLANN
    RadType *radrow,  // arrays((n_cols,), type=RadType): radrow
    Table *dyetrack_weights,  // arrays((n_dyetracks,), type=RadType): All dye weights
    Score *output_scores  // array((n_neighbors,), type=float): returned scores for each neighbor
) {
    for (int nn_i=0; nn_i<n_neighbors; nn_i++) {
        Index neighbor_i = neighbor_dye_iz[nn_i];
        RadType neighbor_dist = neighbor_dists[nn_i];
        WeightType neighbor_weight = *table_get_row(dyetrack_weights, neighbor_i, WeightType);

        output_scores[nn_i] = (Score)(
            neighbor_weight / (0.1 + (neighbor_dist * neighbor_dist))
            // Add a small bias to avoid divide by zero
        );
    }
}


void score_weighted_gaussian_mixture(
    int n_neighbors,
    Size n_cols,
    int *neighbor_dye_iz,  // array((n_neighbors,), type=int): indices to dyetrack
    Table *train_dyemat,  // arrays((n_dyetracks, n_cols), type=RadType): All dye weights
    RadType *radrow,  // arrays((n_cols,), type=RadType): radrow
    Table *dyetrack_weights,  // arrays((n_dyetracks,), type=RadType): All dye weights
    Score *output_scores  // array((n_neighbors,), type=float): returned scores for each neighbor
) {
    double weights[N_MAX_NEIGHBORS];
    double weighted_pdf[N_MAX_NEIGHBORS];
    double weighted_pdf_sum = 0.0;

    // TODO: std_per_dye should be per channel
    // TODO: I need to do an experiment to figure out how
    // to compare lognormal to this model? And ALSO I need to
    // be sure to set that in the _v1 comparison
    double std_per_dye = sqrt(0.1);

    for (int nn_i=0; nn_i<n_neighbors; nn_i++) {
        Index neighbor_i = neighbor_dye_iz[nn_i];
        RadType *neighbor_target_dt = table_get_row(train_dyemat, neighbor_i, RadType);
        WeightType neighbor_weight = *table_get_row(dyetrack_weights, neighbor_i, WeightType);
        weights[nn_i] = (double)neighbor_weight;

        double vdist = (double)0.0;
        double det = 1.0;
        for (Index col_i=0; col_i<n_cols; col_i++) {
            double target_dt_for_col_i = (double)neighbor_target_dt[col_i];
            double delta = target_dt_for_col_i - (double)radrow[col_i];
            double std_units = std_per_dye * (target_dt_for_col_i == 0.0 ? 0.5 : target_dt_for_col_i);
            double variance = std_units * std_units;
            ensure_only_in_debug(variance > 0, "Illegal zero variance");
            det *= variance;
            vdist += delta * delta / variance;
        }
        ensure_only_in_debug(det > 0, "Illegal zero det");
        double inv_sqrt_det = 1.0 / sqrt(det);
        double weight = (double)neighbor_weight * inv_sqrt_det * exp(-vdist / 2.0);
        weighted_pdf[nn_i] = weight;
        weighted_pdf_sum += weight;
    }

    for (int nn_i=0; nn_i<n_neighbors; nn_i++) {
        Score penalty = (Score)(1.0 - exp(-0.8 * weights[nn_i]));
        if(weighted_pdf_sum > 0.0) {
            output_scores[nn_i] = penalty * (Score)(weighted_pdf[nn_i] / weighted_pdf_sum);
        }
        else {
            output_scores[nn_i] = (Score)0;
        }
    }
}

void context_classify_unit_radrows(
    Context *ctx,
    Table unit_radrows,
    Table output_pred_pep_iz,
    Table output_pred_dye_iz,
    Table output_scores
) {
    // unit_radrows, output_*_iz, and output_scores are separated so
    // that they can be passed in in batches.
    // FIND neighbor targets via ANN
    Size n_rows = unit_radrows.n_rows;
    const Size n_neighbors = ctx->n_neighbors;
    ensure(n_neighbors <= N_MAX_NEIGHBORS, "n_neighbors exceeds N_MAX_NEIGHBORS");

    int *neighbor_dye_iz = (int *)alloca(n_rows * n_neighbors * sizeof(int));
    float *neighbor_dists = (float *)alloca(n_rows * n_neighbors * sizeof(float));
    memset(neighbor_dye_iz, 0, n_rows * n_neighbors * sizeof(int));
    memset(neighbor_dists, 0, n_rows * n_neighbors * sizeof(float));

    // FETCH a batch of neighbors from FLANN in one call.
    flann_find_nearest_neighbors_index_float(
        ctx->flann_index_id,
        table_get_row(&unit_radrows, 0, RadType),
        n_rows,
        neighbor_dye_iz,
        neighbor_dists,
        n_neighbors,
        &ctx->flann_params
    );

    for (Index row_i=0; row_i<n_rows; row_i++) {
        RadType *unit_radrow = table_get_row(&unit_radrows, row_i, RadType);
        int *row_neighbor_dye_iz = &neighbor_dye_iz[row_i * n_neighbors];
        Score _output_scores[N_MAX_NEIGHBORS];

        /*
        float *row_neighbor_dists = &neighbor_dists[row_i * n_neighbors];
        score_weighted_inv_square(
            n_neighbors,
            row_neighbor_dye_iz,
            row_neighbor_dists,
            unit_radrow,
            &ctx->train_dyetrack_weights,
            _output_scores
        );
        */

        score_weighted_gaussian_mixture(
            n_neighbors,
            ctx->n_cols,
            row_neighbor_dye_iz,
            &ctx->train_dyemat,
            unit_radrow,
            &ctx->train_dyetrack_weights,
            _output_scores
        );

        // PICK dyetrack winner
        Score highest_score = (Score)0;
        Score score_sum = (Score)0;
        Index highest_score_i = 0;
        for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
            if (_output_scores[nn_i] > highest_score) {
                highest_score = _output_scores[nn_i];
                highest_score_i = nn_i;
            }
            score_sum += _output_scores[nn_i];
        }

        Index most_likely_dye_i = row_neighbor_dye_iz[highest_score_i];
        Score dye_score = highest_score;

        // PICK peptide winner using Maximum Liklihood
        // the .pyx asserts that these are sorted by highest
        // count so we can just pick [0] from the correct dyepep
        Index dyepeps_offset = *table_get_row(&ctx->train_dye_i_to_dyepep_offset, most_likely_dye_i, Index);
        Index *dyepeps_block = table_get_row(&ctx->train_dyepeps, dyepeps_offset, Index);
        ensure_only_in_debug(dyepeps_block[0] == most_likely_dye_i, "dyepeps_block point to wrong block");
        Index most_likely_pep_i = dyepeps_block[1];

        WeightType weight = *table_get_row(&ctx->train_dyetrack_weights, most_likely_dye_i, WeightType);
        Score pep_score = (Score)dyepeps_block[2] / (Score)weight;
        Score score = dye_score * pep_score;

        // Set output
        table_set_row(&output_pred_dye_iz, row_i, &most_likely_dye_i);
        table_set_row(&output_pred_pep_iz, row_i, &most_likely_pep_i);
        table_set_row(&output_scores, row_i, &score);
    }
}


void context_start(Context *ctx) {
    ensure(sanity_check() == 0, "Sanity checks failed");
    ensure(
        ctx->n_neighbors <= ctx->train_dyemat.n_rows,
        "FLANN does not support requesting more neihbors than there are data points"
    );
    ensure(
        ctx->n_neighbors <= N_MAX_NEIGHBORS,
        "Too many neighbors requested"
    );

    // context_print(ctx);

    // CLEAR internally controlled elements
    ctx->flann_params = DEFAULT_FLANN_PARAMETERS;
    ctx->flann_index_id = 0;

    // TODO: Filter low count targets?

    // CREATE the ANN index
    float speedup = 0.0f;
    ctx->flann_index_id = flann_build_index_float(
        table_get_row(&ctx->train_dyemat, 0, RadType),
        ctx->train_dyemat.n_rows,
        ctx->n_cols,
        &speedup,
        &ctx->flann_params
    );

    // TODO: Create inverse variances?

    // TODO: Thread this into batches
    context_classify_unit_radrows(
        ctx,
        ctx->test_unit_radmat,
        ctx->output_pred_pep_iz,
        ctx->output_pred_dye_iz,
        ctx->output_scores
    );
}


void context_free(Context *ctx) {
    if(ctx->flann_index_id) {
        flann_free_index_float(ctx->flann_index_id, &ctx->flann_params);
        ctx->flann_index_id = 0;
    }
}


void context_print(Context *ctx) {
    printf("n_neighbors=%ld\n", ctx->n_neighbors);
    printf("n_cols=%ld\n", ctx->n_cols);
    printf("train_dyemat.n_rows=%ld\n", ctx->train_dyemat.n_rows);
    printf("test_unit_radmat.n_rows=%ld\n", ctx->test_unit_radmat.n_rows);
    for(Index row_i=0; row_i<ctx->test_unit_radmat.n_rows; row_i++) {
        RadType *radrow = table_get_row(&ctx->test_unit_radmat, row_i, RadType);
        for(Index c=0; c<ctx->n_cols; c++) {
            printf("%2.1f ", radrow[c]);
        }
        printf("\n");
    }
}


/*
int test_flann() {

    // TO TEST:
    // Can I build an index with bytes and search with floats?
    //   No: The index and search have to be the same -- flann will actually
    //       run byt it returns bogues results.

    ensure(sanity_check() == 0, "sanity check failed");

    int trows = 2;
    int rows = 3;
    int cols = 2;

    typedef Uint8 TarType;
    TarType *data = (TarType*)calloc(rows * cols, sizeof(TarType));
    data[0 * cols + 0] = (TarType)1;
    data[0 * cols + 1] = (TarType)2;
    data[1 * cols + 0] = (TarType)5;
    data[1 * cols + 1] = (TarType)6;
    data[2 * cols + 0] = (TarType)1;
    data[2 * cols + 1] = (TarType)6;

    TarType *test = (TarType*)calloc(trows * cols, sizeof(TarType));
    test[0 * cols + 0] = (TarType)1.2;
    test[0 * cols + 1] = (TarType)5.5;
    test[1 * cols + 0] = (TarType)4.5;
    test[1 * cols + 1] = (TarType)5.5;

    int nn = 3;
    int *result = (int*)calloc(trows * nn, sizeof(int));
    float *dists = (float*)calloc(trows * nn, sizeof(float));

    struct FLANNParameters p;
    p = DEFAULT_FLANN_PARAMETERS;
    p.algorithm = FLANN_INDEX_KDTREE;
    p.trees = 8;
    p.log_level = FLANN_LOG_INFO;
    p.checks = 64;

    printf("Build index.\n");
    float speedup;
    flann_index_t index_id = flann_build_index_byte(data, rows, cols, &speedup, &p);

    printf("test.\n");
    flann_find_nearest_neighbors_index_byte(index_id, test, trows, result, dists, nn, &p);

    printf("dists=\n");
    for(int i=0; i<trows; i++) {
        for(int j=0; j<nn; j++) {
            printf("%f ", dists[i * nn + j]);
        }
        printf("\n");
    }

    printf("result=\n");
    for(int i=0; i<trows; i++) {
        for(int j=0; j<nn; j++) {
            printf("%d ", result[i * nn + j]);
        }
        printf("\n");
    }

    flann_free_index(index_id, &p);

    free(data);
    free(test);
    free(result);
    free(dists);

    return 0;
}
*/
