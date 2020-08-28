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

Survey

This code's job is to measure how "isolated" each peptide is
with the goal of quickly prediciting how well a given
label/protease scheme will perform.

*/

void dump_dyepeps(SurveyV2FastContext *ctx) {
    Index last_pep_i = 0xFFFFFFFFFFFFFFFF;
    for(Index i=0; i<ctx->dyepeps.n_rows; i++) {
        tab_var(DyePepRec, dyepep, &ctx->dyepeps, i);
        tab_var(Index, mlpep_i, &ctx->dyt_i_to_mlpep_i, dyepep->dtr_i);

        if(last_pep_i != dyepep->pep_i) {
            trace("pep_i%lu\n", dyepep->pep_i);
            last_pep_i = dyepep->pep_i;
        }

        //if(*mlpep_i != dyepep->pep_i) {
            trace("  dyt_i:%-4lu n_reads:%-8lu  mlpep_i:%-4lu   ", dyepep->dtr_i, dyepep->n_reads, *mlpep_i);

            tab_var(DyeType, dyt, &ctx->dyemat, dyepep->dtr_i);
            for (Index k=0; k<ctx->dyemat.n_bytes_per_row; k++) {
                trace("%d ", dyt[k]);
            }
            trace("\n");
        //}
    }
}

void dump_row_of_dyemat(Tab *dyemat, int row, char *prefix) {
    tab_var(DyeType, dyt, dyemat, row);
    for (Index k=0; k<dyemat->n_bytes_per_row; k++) {
        printf("%s%d ", prefix, dyt[k]);
    }
    printf("\n");
}

HashKey pep_i_to_hash_key(Index pep_i) {
    return (pep_i + 1);
}

Index hash_key_pep_i(HashKey pep_i) {
    return (pep_i - 1);
}

void context_pep_measure_isolation(SurveyV2FastContext *ctx, Index pep_i) {
    /*

    Terminology:
        dyt: Dyetrack
        ml: Most Likely
        mic: Most In Contention
        pep: Peptide
        nn: nearest neighbor
        local_dyts: The dyetracks that are associated with pep_i
            (ie "local" because they pertain only to the input parameter)
        global:dyts: The set of all dyetracks in the simulation.
        ml_pep: "Most Likely Peptide" That is the peptide with the most reads
            from any given dyetrack
        self-dyetrack: a dyetrack that has THIS peptide as its ML-pep
        foreign-dyetrack: a dyetrack that has SOME OTHER peptide as its ML-pep
        isolation: A metric of how well separated this peptide is
            from other peptides. This is a relative metric, not
            an actual distance (ie this is NOT a Euclidiean distance)
        contention: The inverse of isolation. A large number means the
            peptide is LESS isolated.

    This function analyzes the "isolation" of a single peptide.
    It has access to:
        * The dyepeps which is a table w/ columns: (dyt_i, pep_i, n_reads)
            That is, each peptide has a list of all dyetracks it can
            create and how many reads that peptide generated for each
            dyetrack.
        * All the dyetracks

    We seek features for this peptide:
         * A measure of "isolation" (bigger number means better isolated)
         * Which OTHER peptide is the most contentious with this peptide?

    Algorithm:
        For this peptide, consider all dyetracks and measure their distance
        to their closest neighbor dyetrack.
        Scale those dyetrack distances by the n_reads for that dyetrack
        Sum all those read-scaled distances up and call that the
        "isolation metric"

        Meanwhile, compute the contention metric for the ml-peptide
        for each dyetrack.
        Sum those ml-pep contentions over every dyetrack
        Find the "most contentious" peptide to return

        In python-like code:

        isolation_by_dyt_i = {}
        contention_by_pep_i = {}
        for (
            dyt_i,
            distance_to_closest_dyt_with_a_foreign_mlpep,
            n_reads_to_dyt_i,
            closest_foreign_mlpep_i
        ) in this_peptide_dyts:
            isolation = n_reads_to_dyt_i * distance_to_closest_dyt_with_a_foreign_mlpep
            contention = n_reads_to_dyt_i / distance_to_closest_dyt_with_a_foreign_mlpep

            isolation_by_dyt_i[dyt_i] += isolation
            contention_by_pep_i[closest_foreign_mlpep_i] += contention

        total_isolation_for_this_pep = sum( isolation_of_dyt_i )
        most_in_contention_pep = the_peptide_with_the_highest_contention_sum(contention_by_pep_i)
    */
	trace("pep_i=%ld\n", pep_i);

    Size n_global_dyts = ctx->n_dyts;
    int n_neighbors = ctx->n_neighbors;
    int dyt_row_n_bytes = ctx->dyemat.n_bytes_per_row;
    int n_dyt_cols = ctx->n_dyt_cols;
    ensure(n_dyt_cols > 0, "no n_dyt_cols");

    // SETUP a local table for the dyepeps OF THIS peptide by using the
    // pep_i_to_dyepep_row_i table to get the start and stop range.
    tab_var(Index, dyepeps_offset_start_of_this_pep, &ctx->pep_i_to_dyepep_row_i, pep_i);
    tab_var(Index, dyepeps_offset_start_of_next_pep, &ctx->pep_i_to_dyepep_row_i, pep_i + 1);
    int _n_local_dyts = *dyepeps_offset_start_of_next_pep - *dyepeps_offset_start_of_this_pep;
    ensure(_n_local_dyts > 0, "no dyts pep_i=%ld (this=%ld next=%ld)", pep_i, *dyepeps_offset_start_of_this_pep, *dyepeps_offset_start_of_next_pep);
    Index n_local_dyts = (Index)_n_local_dyts;

    // Using the pep_i_to_dyepep_row_i we now have the range of the dyepeps and we
    // can create a table subset (which is jsut a view into the table)
    Tab dyepeps = tab_subset(&ctx->dyepeps, *dyepeps_offset_start_of_this_pep, n_local_dyts);

    // We need a contiguous dyemat to feed to the FLANN function so we have
    // to copy each referenced dyemat from the global ctx->dyemat into a local copy.
    // ALLOC a dyemat for all of the dyts of this peptide
    RadType *local_dyemat_buffer = (RadType *)alloca(n_local_dyts * n_dyt_cols * sizeof(DyeType));
    memset(local_dyemat_buffer, 0, n_local_dyts * n_dyt_cols * sizeof(DyeType));
    Tab local_dyemat = tab_by_n_rows(local_dyemat_buffer, n_local_dyts, n_dyt_cols * sizeof(DyeType), TAB_NOT_GROWABLE);

    // LOAD the local dyemat table by copying rows from the global dyemat
    // using the dyt_iz referenced in the dyepeps table.
    for(Index i=0; i<n_local_dyts; i++) {
        tab_var(DyePepRec, dyepep_row, &dyepeps, i);
        tab_var(DyeType, src, &ctx->dyemat, dyepep_row->dtr_i);
        tab_var(DyeType, dst, &local_dyemat, i);
        memcpy(dst, src, dyt_row_n_bytes);
    }

    // Now local_dyemat is a contiguous "local" set of the dyemats that
    // are generated by this pep_i. It must be contiguous so that FLANN
    // and operate on it on one fast call.

    // FLANN needs output buffers to write what it found as the closest neighbors and their distances.
    // ALLOC space for those table on the stack because they shouldn't be too large.
    Size nn_dyt_iz_row_n_bytes = n_neighbors * sizeof(int);
    Size nn_dists_row_n_bytes = n_neighbors * sizeof(float);
    int *nn_dyt_iz_buf = (int *)alloca(n_local_dyts * nn_dyt_iz_row_n_bytes);
    float *nn_dists_buf = (float *)alloca(n_local_dyts * nn_dists_row_n_bytes);
    memset(nn_dyt_iz_buf, 0, n_local_dyts * nn_dyt_iz_row_n_bytes);
    memset(nn_dists_buf, 0, n_local_dyts * nn_dists_row_n_bytes);
    Tab nn_dyt_iz = tab_by_n_rows(nn_dyt_iz_buf, n_local_dyts, nn_dyt_iz_row_n_bytes, TAB_NOT_GROWABLE);
    Tab nn_dists = tab_by_n_rows(nn_dists_buf, n_local_dyts, nn_dists_row_n_bytes, TAB_NOT_GROWABLE);

    // FETCH a batch of neighbors from FLANN in one call against the GLOBAL index of dyetracks
    int ret = flann_find_nearest_neighbors_index_byte(
        ctx->flann_index_id,
        tab_ptr(DyeType, &local_dyemat, 0),
        n_local_dyts,
        nn_dyt_iz_buf,
        nn_dists_buf,
        n_neighbors,
        &ctx->flann_params
    );
    ensure(ret == 0, "flann returned error code");

    // At this point FLANN has found neighbors (and their distances) for each local dyetrack
    // and put the results into:
    // 	  Tab nn_dyt_iz contains the GLBOAL dyt_i index for each neighbor
    // 	  Tab nn_dists contains the distance

	// Now...
	// For each local dyetrack generated by this peptide...
	//    For each neighbor found by FLANN...
	//        Follow that neighbor to its ml-pep.
	//        Often, this ml-pep will be the same as the pep_i in which case it is a "self-dyetrack"
	//        But sometimes this ml-pep will be a different pep in which case it is "foreign"
	//        We only care about the foreign ml-peps

    Size n_neighbors_u = (Size)n_neighbors;
    Size n_reads_total = 0;

	// We need two data-structures as we traverse the local dyts:
	//     isolation_by_dyt_i:   which can be a linear table
    //     contention_by_pep_i:  which needs to be a hash because we can't allocate for every peptide

	tab_alloca(isolation_by_dyt_i, n_local_dyts, sizeof(IsolationType));
	#define N_PEP_HASH_RECS (128)
	Hash contention_by_pep_i = hash_init(alloca(sizeof(HashRec) * N_PEP_HASH_RECS), N_PEP_HASH_RECS);
	Index mlpep_i = 0;

    IsolationType isolation_sum = (IsolationType)0.0;
    for (Index dyt_i=0; dyt_i<n_local_dyts; dyt_i++) {
		// Reminder: dyepeps is the LOCAL dyepeps for pep_i only
        tab_var(DyePepRec, dyepep_row, &dyepeps, dyt_i);

        Index mlpep_i_for_this_dyt_i = tab_get(Index, &ctx->dyt_i_to_mlpep_i, dyepep_row->dtr_i);

		// Get pointers to the nearest neighbor (nn) records (closest neighbot dy_y and
		// distance) that FLANN returned to us for this dyt_i
        tab_var(int, nn_dyt_row_i, &nn_dyt_iz, dyt_i);
        tab_var(float, nn_dists_row_i, &nn_dists, dyt_i);

        int is_local = mlpep_i_for_this_dyt_i == pep_i;

        // Debugging
        trace("  dyt_i:%-4lu  n_reads:%-8lu  mlpep_i:%-4lu  is_local:%1d  ",
            dyepep_row->dtr_i,
            dyepep_row->n_reads,
            mlpep_i_for_this_dyt_i,
            is_local
        );
        tab_var(DyeType, dyt, &ctx->dyemat, dyepep_row->dtr_i);
        for (Index k=0; k<ctx->dyemat.n_bytes_per_row; k++) {
            trace("%d ", dyt[k]);
        }
        trace("\n");

		// For each neighbor of this dyt we want to accumulate a
		// distance measurement -- ONLY IF that neighbor dyetrack's ml-pep foreign
		// (ie that it's ml-pep is some peptide OTHER THAN pep_i).
		// Note, FLANN's distances are Manhattan distances (ie sum of the differences
		// of each dimension). I think this is good enough but it might be worth
		// some sort of sweep over distance metrics.

//		if this dyt is local then theft is those that take away to another pep
//		    iso = prod prob_not_stealing
//		if this dyt is foreign then "rescue" are those that take away to the correct pep
//		    iso = 1 - prod prob_no_rescue


        Index nn_i = 0;
        int global_dyt_i_of_nn_i = 0;

        Float32 p_not_steal_product = 1.0f;
        Float32 p_not_rescue_product = 1.0f;

        for (nn_i=0; nn_i<n_neighbors_u; nn_i++) {
            global_dyt_i_of_nn_i = nn_dyt_row_i[nn_i];
            ensure_only_in_debug(
            	0 <= global_dyt_i_of_nn_i && global_dyt_i_of_nn_i < (int)n_global_dyts,
            	"Illegal dyt in nn lookup: %ld %ld", global_dyt_i_of_nn_i, n_global_dyts
			);

            if((int)dyepep_row->dtr_i == global_dyt_i_of_nn_i) {
                // Do not compare a dyetrack to itself, it will always be zero
                continue;
            }

            // LOOKUP the ml-pep for this dyt_i. Remember, we must use the global_dyt_i_of_nn_i
            // not the local_dyt_i
            mlpep_i = tab_get(Index, &ctx->dyt_i_to_mlpep_i, global_dyt_i_of_nn_i);
            ensure_only_in_debug(0 <= mlpep_i && mlpep_i < ctx->n_peps, "mlpep_i out of bounds %ld %ld", mlpep_i, ctx->n_peps);

            IsolationType distance = (IsolationType)nn_dists_row_i[nn_i];

            // This magic number needs to be parameter swept
            // And the function itself is jsut a guess. There are other
            // distance functions that might be better?
            Float32 k = 0.5f;
            Float32 p_func = 1.0f - expf(-k * distance);

            int mode = 0;
            if(!is_local && mlpep_i == pep_i) {
                // A dyt that is foreign but has a neighbor that is local.
                // This is a potential rescue, we product this into
                // the "p of not rescue"
                p_not_rescue_product *= p_func;
                mode = 1;
            }

            if(is_local && mlpep_i != pep_i) {
                // A dyt that is local has a neighbor that is foreign.
                // This is a potential thief, we product this into
                // the "p of not stealing"
                p_not_steal_product *= p_func;
                mode = 2;
            }

            // Debugging
            if(mode != 0) {
                trace("    nn dyt_i:%-4lu  mlpep_i:%-4lu  dist_to_nn:%7.1f  p_func:%7.5f  mode:%1d  dyt_of_nn:",
                    global_dyt_i_of_nn_i,
                    mlpep_i,
                    distance,
                    p_func,
                    mode
                );
                tab_var(DyeType, dyt, &ctx->dyemat, global_dyt_i_of_nn_i);
                for (Index k=0; k<ctx->dyemat.n_bytes_per_row; k++) {
                    trace("%d ", dyt[k]);
                }
                trace("\n");
            }
        }

        Float32 p_good;
        if(is_local) {
            // A local dyt might be stolen by neighbors
            // What the p(no theft)?
            // p(no theft) = p(nn_0_did_not_steal) * p(nn_1_did_not_steal) ...
            p_good = p_not_steal_product;
        }
        else {
            // A foreign dyt might be rescued by neighbors
            // What the p(any neighbor rescues)?
            // p(at least one rescue) = 1 - p(no rescue)
            // p(no rescue) = p(nn_0_did_not_rescue) * p(nn_1_did_not_rescue) ...

            // Eg: This foreign dyt has two close local neighbors
            // so it has a high chance of being rescued which mean
            // it has a small value for p_not_rescue_product
            // so 1 - small is 1 and thus it has a good isolation
            p_good = 1.0 - p_not_rescue_product;
        }

        IsolationType isolation = dyepep_row->n_reads * p_good;
        isolation_sum += isolation;
        trace("    p_good=%7.5f    isolation=%7.1f \n", p_good, isolation);

        /*
		IsolationType isolation = dyepep_row->n_reads * distance_to_closest_dyt_with_a_foreign_mlpep;
		IsolationType contention = dyepep_row->n_reads / distance_to_closest_dyt_with_a_foreign_mlpep;

		// Set isolation for this dyt
		tab_set(&isolation_by_dyt_i, dyt_i, &isolation);

		// Accumulate into the hash of foreign peps it's contention
		HashKey pep_hash_key = pep_i_to_hash_key(mlpep_i);
		HashRec *by_pep_i_rec = hash_get(contention_by_pep_i, pep_hash_key);
        if(by_pep_i_rec == (HashRec*)0) {
            // hash full!
            ensure(0, "contention_by_pep_i hash table full");
        }
        else if(by_pep_i_rec->key == 0) {
            // New record
            by_pep_i_rec->key = pep_hash_key;
            by_pep_i_rec->contention_val = contention;
        }
        else {
            // Existing record
            by_pep_i_rec->contention_val += contention;
        }

        */
        n_reads_total += dyepep_row->n_reads;
    }

    trace("  \n--> iso_sum=%7.5f  reads=%-7lu  iso=%7.5f\n\n", isolation_sum, n_reads_total, isolation_sum / (IsolationType)n_reads_total);


    // At this point every dyt generated by pep_i has been analyzed
    // and we're ready to make two statements:
    //    What is the total isolation metric of pep_i  (sum over linear table)
    //    What is the closest contentious peptide?     (search for most in hash table)

	// SUM over isolation_by_dyt_i
	/*
	IsolationType total_isolation = (IsolationType)0;
    for (Index dyt_i=0; dyt_i<n_local_dyts; dyt_i++) {
    	total_isolation += tab_get(IsolationType, &isolation_by_dyt_i, dyt_i);
	}
	if(n_reads_total > 0) {
    	total_isolation /= (IsolationType)n_reads_total;
	}
	else {
	    total_isolation = (IsolationType)0.0;
	}
	trace("  total_iso=%f n_reads_total=%ld\n", total_isolation, n_reads_total);

    // FIND the most in contention -- the peptide with the lowest isolation
    IsolationType most_contentious = (IsolationType)0.0;
    Index most_contentious_pep_i = 0;
    for (Index i=0; i<contention_by_pep_i.n_max_recs; i++) {
    	Index pep_i_from_hash = hash_key_pep_i(contention_by_pep_i.recs[i].key);
    	IsolationType contention_from_hash = contention_by_pep_i.recs[i].contention_val;
        if(contention_from_hash > most_contentious) {
            most_contentious = contention_from_hash;
            most_contentious_pep_i = pep_i_from_hash;
        }
    }
	trace("  most_contentious_pep_i=%ld\n", most_contentious_pep_i);

    // RECORD the results into the output tables
    tab_set(&ctx->output_pep_i_to_isolation_metric, pep_i, &total_isolation);
    tab_set(&ctx->output_pep_i_to_mic_pep_i, pep_i, &most_contentious_pep_i);
    */
}


Index context_work_orders_pop(SurveyV2FastContext *ctx) {
    // TODO: This could be dried with similar sim_v2 code
    // (but remember they refer to differnte SurveyV2FastContext structs)
    // NOTE: This return +1! So that 0 can be reserved.
    if(ctx->n_threads > 1) {
        pthread_mutex_lock(&ctx->work_order_lock);
    }

    Index i = ctx->next_pep_i;
    ctx->next_pep_i++;

    if(ctx->n_threads > 1) {
        pthread_mutex_unlock(&ctx->work_order_lock);
    }

    if(i < ctx->n_peps) {
        return i + 1;
    }
    return 0;
}


void *context_work_orders_worker(void *_ctx) {
    // The worker thread. Pops off which pep to work on next
    // continues until there are no more work orders.
    SurveyV2FastContext *ctx = (SurveyV2FastContext *)_ctx;
    while(1) {
        Index pep_i_plus_1 = context_work_orders_pop(ctx);
        if(pep_i_plus_1 == 0) {
            break;
        }
        Index pep_i = pep_i_plus_1 - 1;

        context_pep_measure_isolation(ctx, pep_i);

        if(pep_i % 100 == 0) {
            ctx->progress_fn(pep_i, ctx->n_peps, 0);
        }
    }
    ctx->progress_fn(ctx->n_peps, ctx->n_peps, 0);
    return (void *)0;
}


void context_start(SurveyV2FastContext *ctx) {
//    dump_dyepeps(ctx);

    // Initialize mutex and start the worker thread(s).
    ctx->next_pep_i = 0;

    // TODO: DRY with simialr code in nn_v2

    ensure(
        ctx->n_neighbors <= ctx->dyemat.n_rows,
        "FLANN does not support requesting more neihbors than there are data points"
    );

    // CLEAR internally controlled elements
    ctx->flann_params = DEFAULT_FLANN_PARAMETERS;
    ctx->flann_index_id = 0;

    // CREATE the ANN index
    // TODO: DRY with NN
    float speedup = 0.0f;
    ctx->flann_index_id = flann_build_index_byte(
        tab_ptr(DyeType, &ctx->dyemat, 0),
        ctx->dyemat.n_rows,
        ctx->n_dyt_cols,
        &speedup,
        &ctx->flann_params
    );

    // START threads
    pthread_t ids[256];
    ensure(0 < ctx->n_threads && ctx->n_threads < 256, "Invalid n_threads");

    if(ctx->n_threads > 1) {
        int ret = pthread_mutex_init(&ctx->work_order_lock, NULL);
        ensure(ret == 0, "pthread lock create failed");
    }

    for(Index i=0; i<ctx->n_threads; i++) {
        int ret = pthread_create(&ids[i], NULL, context_work_orders_worker, ctx);
        ensure(ret == 0, "Thread not created.");
    }

    for(Index i=0; i<ctx->n_threads; i++) {
        pthread_join(ids[i], NULL);
    }
}
