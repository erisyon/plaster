#include "_sub_pixel_align.h"
#include "alloca.h"
#include "c_common.h"
#include "math.h"
#include "memory.h"
#include "pthread.h"
#include "stdarg.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "unistd.h"

void _dump_vec(Float64 *vec, int width, int height, char *msg) {
  trace("VEC %s [\n", msg);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      fprintf(_log, "%4.4f, ", vec[y * width + x]);
    }
    fprintf(_log, "\n");
  }
  fprintf(_log, "]\n");
}

void _slice(F64Arr *im, Index row_i, Index n_rows_per_slice, Float64 *out_slice,
            Size width) {
  // Sum over the vertical element to go from a short 2D signal
  // to a 1D signal.

  memset(out_slice, 0, sizeof(Float64) * width);
  for (Index i = 0; i < n_rows_per_slice; i++) {
    Float64 *src = f64arr_ptr1(im, row_i + i);
    Float64 *dst = out_slice;
    for (Index col_i = 0; col_i < width; col_i++) {
      *dst++ += *src++;
    }
  }
}

void _cubic_spline_segment(Float64 p0, Float64 p1, Float64 p2, Float64 p3,
                           Float64 *dst, Size n_steps) {
  // Catmull-Rom spline
  //
  // p0 --------- p1 ----------- p2 ----------- p3
  //              We are filling
  //              in this section.
  //

  Float64 cubic = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
  Float64 quadratic = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
  Float64 linear = -0.5 * p0 + 0.5 * p2;
  Float64 constant = p1;

  Float64 step = 1.0 / n_steps;
  Float64 x = 0.0;
  for (Index x_i = 0; x_i < n_steps; x_i++) {
    Float64 x2 = x * x;
    Float64 x3 = x * x2;
    *dst++ = cubic * x3 + quadratic * x2 + linear * x + constant;
    x += step;
  }
}

void _rescale(Float64 *slice, Float64 *out_slice, Size width, Size scale) {
  // Use cubic resampling to expand from slice to out_slice
  // which is scale times larger.
  // The spline interplition uses the derivate of successive points
  // to estimate the splce. This means that the first and last two points
  // have to be treated specially -- specifically by duplicating the edge
  // values.

  // Interpolate the first point by duplicating point[0]
  _cubic_spline_segment(slice[0], slice[0], slice[1], slice[2], &out_slice[0],
                        scale);

  // Interpolate the middle points (no boundary effects)
  for (Index i = 1; i < width - 2; i++) {
    _cubic_spline_segment(slice[i - 1], slice[i], slice[i + 1], slice[i + 2],
                          &out_slice[i * scale], scale);
  }

  // Interpolate the last two points duplicating the point[-1] twice.
  Index i = width - 2;
  _cubic_spline_segment(slice[i - 1], slice[i], slice[i + 1], slice[i + 1],
                        &out_slice[i * scale], scale);

  i = width - 1;
  _cubic_spline_segment(slice[i - 1], slice[i], slice[i + 1], slice[i + 1],
                        &out_slice[i * scale], scale);
}

int _convolve(Float64 *cy0, Float64 *cyi, int scale, int width) {
  // Shift cyi relateive to cy0 and sum the product of the intersection.
  // The inner-most loop and should be the fastest.

  Float64 max_sum = 0.0;
  int max_offset = 0;

  // The widht needs to be constant so that all offset functions
  // end up with the same number of compares (otherwise you have normalize
  // the comparison)
  int _width = width - scale;

  // SCAN offset, compute sum(a * shifted(b)) and track the maximum
  // to find the best fit.
  for (int offset = -scale; offset <= scale; offset++) {
    Float64 *_cy0 = cy0;
    Float64 *_cyi = cyi;

    // TRIM appropriate side of the function
    if (offset < 0) {
      _cyi = &cyi[-offset];
    } else {
      _cy0 = &cy0[offset];
    }

    Float64 sum = 0.0;
    for (Index i = 0; i < _width; i++) {
      sum += (*_cy0++) * (*_cyi++);
    }
    if (sum > max_sum) {
      max_sum = sum;
      max_offset = offset;
    }
  }

  return -1 * max_offset; // Minus to get it into the original coords.
}

char *sub_pixel_align_one_cycle(SubPixelAlignContext *ctx, Index cy_i) {
  // The work horse
  // Loops over each slice and "convolves" each slice re-using the
  // same buffer on each slice so that the memory requirements don't go up.

  Size height = ctx->mea_h;
  Size width = ctx->mea_w;
  Size scale = ctx->scale;
  Size slice_h = ctx->slice_h;
  Size n_slices = ctx->_n_slices;
  Size large_width = width * scale;

  int *offset_samples = (int *)malloc(sizeof(int) * n_slices);
  Float64 *slice_buffer = (Float64 *)malloc(sizeof(Float64) * width);
  Float64 *large_slice_buffer =
      (Float64 *)malloc(sizeof(Float64) * large_width);

  ensure(offset_samples != NULL, "malloc failed");
  ensure(slice_buffer != NULL, "malloc failed");
  ensure(large_slice_buffer != NULL, "malloc failed");

  F64Arr cy_im = f64arr_subset(&ctx->cy_ims, cy_i);
  for (Index slice_i = 0; slice_i < n_slices; slice_i++) {
    Index row_i = slice_i * slice_h;
    _slice(&cy_im, row_i, slice_h, slice_buffer, width);
    _rescale(slice_buffer, large_slice_buffer, width, scale);
    Float64 *large_cy0_slice = f64arr_ptr1(&ctx->_large_cy0_slices, slice_i);

    // Compare cycle i slice with the paired cycle 0 slice.
    Index offset =
        _convolve(large_cy0_slice, large_slice_buffer, scale, large_width);
    offset_samples[slice_i] = offset;
  }

  // We may need filtering, for now just compute a mean
  Float64 mean_offset = 0.0;
  for (Index slice_i = 0; slice_i < n_slices; slice_i++) {
    mean_offset += offset_samples[slice_i];
  }
  mean_offset /= (Float64)n_slices;
  Float64 ret = mean_offset / scale;
  *f64arr_ptr1(&ctx->out_offsets, cy_i) = ret;

  free(large_slice_buffer);
  free(slice_buffer);
  free(offset_samples);

  return NULL;
}

char *context_init(SubPixelAlignContext *ctx) {
  // The cycle 0 slices are used over and over again so we
  // cut them up and rescale them once and re-use them even through
  // this burns a lot memory.

  Size height = ctx->mea_h;
  Size width = ctx->mea_w;
  Size scale = ctx->scale;
  Size slice_h = ctx->slice_h;
  Size n_slices = height / slice_h;
  ctx->_n_slices = n_slices;

  Float64 *slice_buffer = (Float64 *)malloc(sizeof(Float64) * width);
  ensure(slice_buffer != NULL, "malloc failed");
  Size cy0_slices_shape[2] = {n_slices, scale * width};
  ctx->_large_cy0_slices = f64arr_malloc(2, cy0_slices_shape);

  F64Arr cy0_im = f64arr_subset(&ctx->cy_ims, 0);
  for (Index slice_i = 0; slice_i < n_slices; slice_i++) {
    Index row_i = slice_i * slice_h;
    _slice(&cy0_im, row_i, slice_h, slice_buffer, width);
    _rescale(slice_buffer, f64arr_ptr1(&ctx->_large_cy0_slices, slice_i), width,
             scale);
  }

  free(slice_buffer);

  return NULL;
}

char *context_free(SubPixelAlignContext *ctx) {
  f64arr_free(&ctx->_large_cy0_slices);
  return NULL;
}
