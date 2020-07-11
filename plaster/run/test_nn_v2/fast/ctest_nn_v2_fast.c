#include "flann.h"
#include "stdio.h"
#include "stdlib.h"


int test_flann() {
  int trows = 2;
  int rows = 3;
  int cols = 2;

  float *data = (float*)calloc(rows * cols, sizeof(float));
  data[0 * cols + 0] = 1.0;
  data[0 * cols + 1] = 2.0;
  data[1 * cols + 0] = 5.0;
  data[1 * cols + 1] = 6.0;
  data[2 * cols + 0] = 1.0;
  data[2 * cols + 1] = 6.0;

  float *test = (float*)calloc(trows * cols, sizeof(float));
  test[0 * cols + 0] = 1.2;
  test[0 * cols + 1] = 5.5;
  test[1 * cols + 0] = 4.5;
  test[1 * cols + 1] = 5.5;

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
  flann_index_t index_id = flann_build_index(data, rows, cols, &speedup, &p);

  printf("test.\n");
  flann_find_nearest_neighbors_index(index_id, test, trows, result, dists, nn, &p);

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
