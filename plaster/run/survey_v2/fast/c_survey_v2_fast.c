#include "stdint.h"
#include "alloca.h"
#include "stdio.h"
#include "stdlib.h"
#include "stdarg.h"
#include "memory.h"
#include "pthread.h"
#include "unistd.h"
#include "math.h"
#include "c_survey_v2_fast.h"


/*

Looks like there's more and more common c code
Need to get a common pxd file
Change all dye_ and dt stuff to consistent dyt_
I really want more than one version of table_init

mlpep = maximum likely peptide

Build a "dyt_to_mlpep" LUT

for each peptide (parallel)
    extract out all of the dyts for that peptide (a groupby in pandas or a index similar to sim_v2)
        Lookup n_neighbors for each of those dyetracks
        that's a matrix of n_dyt, n_neighbor
        there's a parallel dist vector returned from flann
        replace each value in that table with a LUT from dyt_to_mlpep
        Search along the columns for the first pep that isn't THIS pep
        That pep is the closest interference pep

        Multiply f(dist) * frac_of_reads_to_this_dyt
        Add all those up and that's the isolation metric. A big number is better.
*/

void context_measure_peptide_isolation(
    Context *ctx,
    Index pep_i,
    Table dyt_i_to_mlpep_i
) {
    int n_neighbors = ctx->?;
    int n_dyts = ctx->?;
    int dyt_row_n_bytes = ctx->?;

    Index *dyt_iz = ?; // A pointer to an array of ints that are indexes into the dyemat rows

    DyeType *local_dyemat_buffer = (DyeType *)alloca(dyt_row_n_bytes * n_dyts);
    Table local_dyemat = table_init(local_dyemat_buffer, dyt_row_n_bytes * n_dyts, dyt_row_n_bytes);

    for(Index i=0; i<n_dyts; i++) {
        Index dyt_i = dyt_iz[i];
        DyeType *src_dyt_ptr = table_get_row(ctx->dyemat, dyt_i, DyeType)
        DyeType *dst_dyt_ptr = table_get_row(local_dyemat, i, DyeType)
        memcpy(dst_dyt_ptr, src_dyt_ptr, dyt_row_n_bytes);
    }

    int *neighbor_dye_iz = (int *)alloca(n_dyt * n_neighbors * sizeof(int));
    float *neighbor_dists = (float *)alloca(n_dyt * n_neighbors * sizeof(float));
    memset(neighbor_dye_iz, 0, n_dyt * n_neighbors * sizeof(int));
    memset(neighbor_dists, 0, n_dyt * n_neighbors * sizeof(float));

    // FETCH a batch of neighbors from FLANN in one call.
    flann_find_nearest_neighbors_index_float(
        ctx->flann_index_id,
        local_dyemat_buffer,
        n_dyts,
        neighbor_dye_iz,
        neighbor_dists,
        n_neighbors,
        &ctx->flann_params
    );

    /*
    Lookup n_neighbors for each of those dyetracks
    that's a matrix of n_dyt, n_neighbor
    there's a parallel dist vector returned from flann
    replace each value in that table with a LUT from dyt_to_mlpep
    Search along the columns for the first pep that isn't THIS pep
    That pep is the closest interference pep

    Multiply f(dist) * frac_of_reads_to_this_dyt
    Add all those up and that's the isolation metric. A big number is better.
    */

    Index contention_pep_i = 0;
    Float32 contention_pep_dist = 0;
    for (Index i=0; i<n_dyts; i++) {

        Make a per row lookup for below
        table_

        for (Index nn_i=0; nn_i<n_neighbors; nn_i++) {
            Index nn_dye_i = neighbor_dye_iz[nn_i];
            ensure_only_in_debug(0 <= nn_dye_i && nn_dye_i < n_dyts);

            mlpep_i = dyt_i_to_mlpep_i[nn_dye_i];
            ensure_only_in_debug(0 <= mlpep_i && mlpep_i < n_peps);

            if(ml_pep_i != pep_i) {
                // Found the first dyt with a ML peptide that isn't this peptide
                Record it's distance
                contention_pep_i = ml_pep_i;
                contention_pep_dist = neighbor_dists[]
                break
            }
        }
        if(nn_i == n_neighbors) {
            // Got to end without finding another peptide in the neighbor list
        }
    }
}



void context_start(Context *ctx) {
}

