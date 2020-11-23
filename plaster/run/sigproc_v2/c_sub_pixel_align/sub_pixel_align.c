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
#include "_sub_pixel_align.h"


void _slices(SubPixelAlignContext *ctx, Index cy_i, F64Arr *out_slices) {
    // Cut the cycle image up into 1D slices (summed over vertical)

    Size im_bot = ctx->mea_h - ctx->slice_h;
    Size w = ctx->mea_w;

    memset(out_slice, 0, sizeof(PixType) * w);

    for(Index top=0; top<im_bot; top++) {
        Index bot = min(top + ctx->slice_h, im_bot);
        for(Index row_i=top; row_i<bot; row_i++) {
            PixType *src = f64arr_ptr2(ctx->cy_ims, cy_i, row_i);
            PixType *dst = out_slice;
            for(Index col_i=0; col_i<w; col_i++) {
                *dst++ += *src++;
            }
        }
    }
}


void _rescale(SubPixelAlignContext *ctx, PixType *slice) {
}


char *sub_pixel_align_one_cycle(SubPixelAlignContext *ctx, Index cy_i) {
}


char *context_init(SubPixelAlignContext *ctx) {

    // SLICE up cycle 0 and re-use it on each cycle
    F64Arr *cy0_slices = ?

    _slices(SubPixelAlignContext *ctx, 0, cy0_slices);


    return NULL;
}

char *context_free(SubPixelAlignContext *ctx) {
    return NULL;
}

