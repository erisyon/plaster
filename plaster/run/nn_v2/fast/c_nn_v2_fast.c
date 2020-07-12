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


typedef Float32 RadType;


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
    RadType *dyetracks,  // arrays((n_dyetracks, n_cols), type=RadType): All dyetracks
    RadType *dyetrack_weights,  // arrays((n_dyetracks,), type=RadType): All dye weights
    Score *output_scores  // array((n_neighbors,), type=float): returned scores for each neighbor
) {
    // scoring funcs take neighors and distances and return scores for each
    for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
        Index neighbor_i = neighbor_iz[nn_i];
        RadType neighbor_dist = neighbor_dists[nn_i];
        RadType neighbor_weight = dyetrack_weights[neightbor_i];
        output_scores[nn_i] = (Score)(neighbor_weight / (neighbor_dist * neighbor_dist));
    }
}


void context_classify_radrows(NNContext *ctx, Size n_rows, RadType *radrows, CallRec *output_calls) {
    // FIND neighbor targets via ANN
    const int n_cols = ctx->n_cols;
    const int n_neighbors = ctx->n_neighbors;
    ensure(n_neighbors < N_MAX_NEIGHBORS, "n_neighbors exceeds N_MAX_NEIGHBORS");
    int *neighbor_iz = (int*)alloca(n_rows * n_neighbors * sizeof(int));
    float *neighbor_dists = (float*)alloca(n_rows * n_neighbors, sizeof(float));

    flann_find_nearest_neighbors_index_float(
        ctx->flann_index_id,
        radrows,
        n_rows,
        neighbor_iz,
        neighbor_dists,
        n_neighbors,
        &ctx->flann_params
    );

    for (Index row_i=0; row_i<n_rows; row_i++) {
        RadType *radrow = &radrows[row_i];

        int *row_neighbor_iz = &neighbor_iz[row_i * n_neighbors];
        float *row_neighbor_dists = &neighbor_dists[row_i * n_neighbors];

        Score output_scores[N_MAX_NEIGHBORS];
        score_weighted_inv_square(
            n_neighbors,
            row_neighbor_iz,
            row_neighbor_dists,
            radrow,
            ctx->dyetracks_as_radtype,
            ctx->dyetrack_weights,
            output_scores
        );

        // PICK winner
        Score score_sum = (Score)0;
        Score highest_score = (Score)0;
        Index highest_score_i = 0;
        for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
            if(output_scores[nn_i] > highest_score) {
                highest_score = output_scores[nn_i];
                highest_score_i = nn_i;
            }
            score_sum += output_scores[nn_i];
        }

        // Set output
        output_calls[row_i].dt_i = row_neighbor_iz[highest_score_i];
        output_calls[row_i].score = highest_score;
    }
}


void context_work_orders_start(NNContext *ctx) {
    ctx->dyetracks_as_floats = (void *)0;
    ctx->flann_index_id = 0;
    ctx->flann_params = DEFAULT_FLANN_PARAMETERS;

    // TODO: Filter low count targets

    // TYPECAST the dyetracks to floats
    ctx->dyetracks_as_radtype = (RadType *)calloc(ctx->train_dyetracks_n_rows * ctx->n_cols, sizeof(RadType))
    DyeType *src = ctx->train_dyetracks;
    RadType *dst = ctx->dyetracks_as_radtype;
    Size n = ctx->train_dyetracks_n_rows * ctx->n_cols;
    for(Index i=0; i<n; i++) {
        *dst = (RadType)*src;
        dst++;
        src++;
    }

    ctx->dyetrack_weights = (RadType *)calloc(ctx->train_dyetracks_n_rows, sizeof(RadType));
    for(Index i=0; i<ctx->train_dyepeps_n_rows; i++) {
        // TODO: This potentially loses precision since dyetrack_weights is a float32 and count is 64
        DyePepRec *dyepeprec = &ctx->train_dyepeps[i];
        ctx->dyetrack_weights[dyepeprec->dtr_i] += (RadType)dyepeprec->count;
    }

    // CREATE the ANN index
    float speedup = 0.0f;
    ctx->flann_index_id = flann_build_index_float(
        ctx->dyetracks_as_radtype,
        ctx->train_dyetracks_n_rows,
        ctx->n_cols,
        &speedup,
        &ctx->flann_params
    );

    // TODO: Create inverse variances?

    // TODO: Thread this into batches
    Radrow *radrow = ctx->radmat;
    context_classify_radrows(ctx, radrow);
}

void context_free(NNContext *ctx) {
    if(ctx->dyetracks_as_floats) {
        free(ctx->dyetracks_as_floats);
        ctx->dyetracks_as_floats = (void *)0;
    }
    if(ctx->flann_index_id) {
        flann_free_index_float(ctx->flann_index_id, &ctx->flann_params);
        ctx->flann_index_id = 0;
    }
}



int sanity_check() {
    printf("SANITY......................................\n");
    if(sizeof(Float32) != 4) {
        printf("Failed sanity check: sizeof Float32\n");
        return 1;
    }

    return 0;
}

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
