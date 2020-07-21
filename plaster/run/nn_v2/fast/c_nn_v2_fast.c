#include "stdint.h"
#include "alloca.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdarg.h"
#include "memory.h"
#include "pthread.h"
#include "unistd.h"
#include "c_nn_v2_fast.h"


void ensure(int expr, const char *fmt, ...) {
    // Replacement for assert with var-args and local control of compilation.
    // See ensure_only_in_debug below.
    va_list args;
    va_start(args, fmt);
    if(!expr) {
        vfprintf(stderr, fmt, args);
        fflush(stderr);
        exit(1);
    }
    va_end(args);
}


#ifdef DEBUG
    #define ensure_only_in_debug ensure
#else
    #define ensure_only_in_debug(...) ((void)0)
#endif


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
    int *neighbor_iz,  // array((n_neighbors,), type=int): indices to dyetrack
    float *neighbor_dists,  // array((n_neighbors,), type=float): distances computed by FLANN
    RadType *radrow,  // arrays((n_cols,), type=RadType): radrow
    RadType *dyemat,  // arrays((n_dyetracks, n_cols), type=RadType): All dyetracks
    WeightType *dyetrack_weights,  // arrays((n_dyetracks,), type=RadType): All dye weights
    Score *output_scores  // array((n_neighbors,), type=float): returned scores for each neighbor
) {
    // scoring funcs take neighors and distances and return scores for each
    for (int nn_i=0; nn_i<n_neighbors; nn_i++) {
        Index neighbor_i = neighbor_iz[nn_i];
        RadType neighbor_dist = neighbor_dists[nn_i];
        WeightType neighbor_weight = dyetrack_weights[neighbor_i];

        output_scores[nn_i] = (Score)(
            neighbor_weight / (0.1 + (neighbor_dist * neighbor_dist))
            // Add a small bias to avoid divide by zero
        );
        printf("neighbor_dist=%f\n", neighbor_dist);
        printf("neighbor_weight=%f\n", neighbor_weight);
        printf("output_scores[nn_i]=%f\n", output_scores[nn_i]);
    }
}


void context_classify_unit_radrows(Context *ctx, Size n_rows, RadType *unit_radrows, Index32 *output_pred_iz, Score *output_scores) {
    // radrows, output_pred_iz, and output_scores are seprated so
    // that they can be passed in in batches.

    // FIND neighbor targets via ANN
    const Size n_cols = ctx->n_cols;
    const Size n_neighbors = ctx->n_neighbors;
    ensure(n_neighbors <= N_MAX_NEIGHBORS, "n_neighbors exceeds N_MAX_NEIGHBORS");

    int *neighbor_iz = (int *)alloca(n_rows * n_neighbors * sizeof(int));
    float *neighbor_dists = (float *)alloca(n_rows * n_neighbors * sizeof(float));
    memset(neighbor_iz, 0, n_rows * n_neighbors * sizeof(int));
    memset(neighbor_dists, 0, n_rows * n_neighbors * sizeof(float));

    // FETCH a batch of neighbors from FLANN in one call.
    flann_find_nearest_neighbors_index_float(
        ctx->flann_index_id,
        unit_radrows,
        n_rows,
        neighbor_iz,
        neighbor_dists,
        n_neighbors,
        &ctx->flann_params
    );

    for (Index row_i=0; row_i<n_rows; row_i++) {
        RadType *unit_radrow = &unit_radrows[row_i * n_cols];
        int *row_neighbor_iz = &neighbor_iz[row_i * n_neighbors];
        float *row_neighbor_dists = &neighbor_dists[row_i * n_neighbors];

        Score output_scores[N_MAX_NEIGHBORS];
        score_weighted_inv_square(
            n_neighbors,
            row_neighbor_iz,
            row_neighbor_dists,
            unit_radrow,
            ctx->train_dyemat,
            ctx->train_dyetrack_weights,
            output_scores
        );

        // PICK winner
        Score score_sum = (Score)0;
        Score highest_score = (Score)0;
        Index highest_score_i = 0;
        for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
            if (output_scores[nn_i] > highest_score) {
                highest_score = output_scores[nn_i];
                highest_score_i = nn_i;
            }
            score_sum += output_scores[nn_i];
        }

        // Set output
        output_pred_iz[row_i] = row_neighbor_iz[highest_score_i];
        output_scores[row_i] = highest_score;
        printf("row_i=%ld\n", row_i);

        // TODO: In current state there's a stack corruption around here
        // because I';; sometimes get a stack smashing detected message
        // I need to switch over to using protected tables like sim_v2

        //float a = output_scores[row_i];
        //printf("output_scores[row_i]=%2.1f\n", output_scores[row_i]);
    }
}


void context_start(Context *ctx) {
    ensure(sanity_check() == 0, "Sanity checks failed");
    ensure(
        ctx->n_neighbors <= ctx->train_dyemat_n_rows,
        "FLANN does not support requesting more neihbors than there are data points"
    );

    context_print(ctx);

    // CLEAR internally controlled elements
    ctx->flann_params = DEFAULT_FLANN_PARAMETERS;
    ctx->flann_index_id = 0;

    // TODO: Filter low count targets?

    // CREATE the ANN index
    float speedup = 0.0f;
    ctx->flann_index_id = flann_build_index_float(
        ctx->train_dyemat,
        ctx->train_dyemat_n_rows,
        ctx->n_cols,
        &speedup,
        &ctx->flann_params
    );

    // TODO: Create inverse variances?

    // TODO: Thread this into batches
    context_classify_unit_radrows(ctx, ctx->test_unit_radmat_n_rows, ctx->test_unit_radmat, ctx->output_pred_iz, ctx->output_scores);
}


void context_free(Context *ctx) {
    if(ctx->flann_index_id) {
        flann_free_index_float(ctx->flann_index_id, &ctx->flann_params);
        ctx->flann_index_id = 0;
    }
}


void context_print(Context *ctx) {
//    RadType *train_dyemat;
//    WeightType *train_dyetrack_weights;
//    Index32 *output_pred_iz;
//    Score *output_scores;

    printf("n_neighbors=%ld\n", ctx->n_neighbors);
    printf("n_cols=%ld\n", ctx->n_cols);
    printf("train_dyemat_n_rows=%ld\n", ctx->train_dyemat_n_rows);
    printf("test_unit_radmat_n_rows=%ld\n", ctx->test_unit_radmat_n_rows);
    for(Index i=0; i<ctx->test_unit_radmat_n_rows; i++) {
        for(Index c=0; c<ctx->n_cols; c++) {
            printf("%2.1f ", ctx->test_unit_radmat[i*ctx->n_cols + c]);
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
