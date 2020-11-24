#ifndef C_COMMON_NEW_H
#define C_COMMON_NEW_H

// F64Arr
//----------------------------------------------------------------------------------------

#define MAX_ARRAY_DIMS (4)

typedef struct {
    Float64 *base;
    Size n_dims;

    // Shape is the number of elements in each dimensions
    // or zero if none.
    Size shape[MAX_ARRAY_DIMS];

    // pitch is the product of all subordinate shapes
    // (ie, the amount you need to add to an index of that
    // dimension to get to the next element).
    Size pitch[MAX_ARRAY_DIMS];
} F64Arr;

void f64arr_set_shape(F64Arr *arr, Size n_dims, Size *shape);
F64Arr f64arr(void *base, Size n_dims, Size *shape);
F64Arr f64arr_subset(F64Arr *src, Index i);
F64Arr f64arr_malloc(Size n_dims, Size *shape);
void f64arr_free(F64Arr *arr);

Float64 *f64arr_ptr1(F64Arr *arr, Index i);
Float64 *f64arr_ptr2(F64Arr *arr, Index i, Index j);
Float64 *f64arr_ptr3(F64Arr *arr, Index i, Index j, Index k);
Float64 *f64arr_ptr4(F64Arr *arr, Index i, Index j, Index k, Index l);


#endif