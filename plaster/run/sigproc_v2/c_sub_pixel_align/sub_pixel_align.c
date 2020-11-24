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


void _dump_vec(Float64 *vec, int width, int height, char *msg) {
    trace("VEC %s [\n", msg);
    for(int y=0; y<height; y++) {
        for(int x=0; x<width; x++) {
            fprintf(_log, "%4.4f, ", vec[y*width + x]);
        }
        fprintf(_log, "\n");
    }
    fprintf(_log, "]\n");
}


void _slice(F64Arr *im, Index row_i, Index n_rows_per_slice, Float64 *out_slice, Size width) {
    memset(out_slice, 0, sizeof(Float64) * width);
    for(Index i=0; i<n_rows_per_slice; i++) {
        Float64 *src = f64arr_ptr1(im, row_i + i);
        Float64 *dst = out_slice;
        for(Index col_i=0; col_i<width; col_i++) {
            *dst++ += *src++;
        }
    }
}


void _cubic_spline_segment(Float64 p0, Float64 p1, Float64 p2, Float64 p3, Float64 *dst, Size n_steps) {
    // Catmull-Rom spline
    Float64 cubic     = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3;
    Float64 quadratic =      p0 - 2.5*p1 + 2.0*p2 - 0.5*p3;
    Float64 linear    = -0.5*p0          + 0.5*p2;
    Float64 constant  =               p1;

    Float64 step = 1.0 / n_steps;
    Float64 x = 0.0;
    for(Index x_i=0; x_i<n_steps; x_i++) {
        Float64 x2 = x * x;
        Float64 x3 = x * x2;
        *dst++ = cubic * x3 + quadratic * x2 + linear * x + constant;
        x += step;
    }
}


void _rescale(
    Float64 *slice,
    Float64 *out_slice,
    Size width,
    Size scale
) {
    // Use cubic resampling to expand from slice to out_slice which is scale times larger.

    // Interpolate the first point
    _cubic_spline_segment(
        slice[0], slice[0], slice[1], slice[2],
        &out_slice[0],
        scale
    );

    // Interpolate the middle points
    for(Index i=1; i<width-2; i++) {
        _cubic_spline_segment(
            slice[i-1], slice[i], slice[i+1], slice[i+2],
            &out_slice[i * scale],
            scale
        );
    }

    // Interpolate the last two points
    Index i = width - 2;
    _cubic_spline_segment(
        slice[i-1], slice[i], slice[i+1], slice[i+1],
        &out_slice[i * scale],
        scale
    );

    i = width - 1;
    _cubic_spline_segment(
        slice[i-1], slice[i], slice[i+1], slice[i+1],
        &out_slice[i * scale],
        scale
    );
}


int _convolve(Float64 *cy0, Float64 *cyi, int scale, int width) {
    // Shift cyi relative to cy0, so a negative offset means
    // that cyi is to the left of cy0

    Float64 max_sum = 0.0;
    int max_offset = 0;
    int _width = width - scale;
    for(int offset = -scale; offset <= scale; offset++) {
        Float64 *_cy0 = cy0;
        Float64 *_cyi = cyi;

        if(offset < 0) {
            _cyi = &cyi[-offset];
        }
        else {
            _cy0 = &cy0[offset];
        }

        Float64 sum = 0.0;
        for(Index i=0; i<_width; i++) {
            sum += (*_cy0++) * (*_cyi++);
        }
        if(sum > max_sum) {
            max_sum = sum;
            max_offset = offset;
        }
    }

    return -1 * max_offset; // Minus to get it into the original coords.
}


char *sub_pixel_align_one_cycle(SubPixelAlignContext *ctx, Index cy_i) {
    Size height = ctx->mea_h;
    Size width = ctx->mea_w;
    Size scale = ctx->scale;
    Size slice_h = ctx->slice_h;
    Size n_slices = ctx->_n_slices;
    Size large_width = width * scale;

    int *offset_samples = (int *)malloc(sizeof(int) * n_slices);
    Float64 *slice_buffer = (Float64 *)malloc(sizeof(Float64) * width);
    Float64 *large_slice_buffer = (Float64 *)malloc(sizeof(Float64) * large_width);

    F64Arr cy_im = f64arr_subset(&ctx->cy_ims, cy_i);
    for(Index slice_i=0; slice_i<n_slices; slice_i++) {
        Index row_i = slice_i * slice_h;
        _slice(&cy_im, row_i, slice_h, slice_buffer, width);
        _rescale(slice_buffer, large_slice_buffer, width, scale);
        Float64 *large_cy0_slice = f64arr_ptr1(&ctx->_large_cy0_slices, slice_i);

        Index offset = _convolve(large_cy0_slice, large_slice_buffer, scale, large_width);
        offset_samples[slice_i] = offset;
    }

    // We may need filtering, for now just compute a mean
    Float64 mean_offset = 0.0;
    for(Index slice_i=0; slice_i<n_slices; slice_i++) {
        mean_offset += offset_samples[slice_i];
    }
    mean_offset /= (Float64)n_slices;
    Float64 ret = 2 * mean_offset / scale;
        // Not sure where this factor of 2 comes from but it
        // is clear from testing that it is needed. Presumably
        // because we search both side of the offset
    *f64arr_ptr1(&ctx->out_offsets, cy_i) = ret;

    free(large_slice_buffer);
    free(slice_buffer);
    free(offset_samples);

    return NULL;
}


char *context_init(SubPixelAlignContext *ctx) {
    // SLICE up cycle 0 and re-use it on each cycle
    Size height = ctx->mea_h;
    Size width = ctx->mea_w;
    Size scale = ctx->scale;
    Size slice_h = ctx->slice_h;
    Size n_slices = height / slice_h;
    ctx->_n_slices = n_slices;

    Float64 *slice_buffer = (Float64 *)malloc(sizeof(Float64) * width);
    Size cy0_slices_shape[2] = { n_slices, scale * width };
    ctx->_large_cy0_slices = f64arr_malloc(2, cy0_slices_shape);

    F64Arr cy0_im = f64arr_subset(&ctx->cy_ims, 0);
    for(Index slice_i=0; slice_i<n_slices; slice_i++) {
        Index row_i = slice_i * slice_h;
        _slice(&cy0_im, row_i, slice_h, slice_buffer, width);
        _rescale(slice_buffer, f64arr_ptr1(&ctx->_large_cy0_slices, slice_i), width, scale);
    }

    free(slice_buffer);

    return NULL;
}


char *context_free(SubPixelAlignContext *ctx) {
    f64arr_free(&ctx->_large_cy0_slices);
    return NULL;
}
