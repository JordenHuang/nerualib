#include <stdio.h>

#define NERUALIB_IMPLEMENTATION
#include "neuralib.h"


int main(void)
{
    nl_rand_init(0, 0);
#if 0
    size_t r = 2;
    size_t c = 2;
    Mat m = nl_mat_alloc(1, c);
    Mat n = nl_mat_alloc(r, 1);
    Mat dst = nl_mat_alloc(1, 1);

    // Mat rand
    nl_mat_rand(m);
    nl_mat_print(m);

    // // Mat one
    // nl_mat_one(m);
    // nl_mat_print(m);

    // // Mat zero
    // nl_mat_zero(m);
    // nl_mat_print(m);

    printf("----------\n");
    // Mat add
    // nl_mat_rand(n);
    // nl_mat_print(n);
    // nl_mat_add(dst, m, n);
    // nl_mat_print(dst);


    printf("---------- Dot:\n");
    // Mat dot
    nl_mat_one(n);
    nl_mat_print(n);
    nl_mat_dot(dst, m, n);
    nl_mat_print(dst);

    printf("----------\n");
    // Mat multiply constant
    nl_mat_print(m);
    nl_mat_mult_c(m, m, 2.f);
    nl_mat_print(m);

    printf("----------\n");
    // Mat transpose
    // nl_mat_free(m);
    // m = nl_mat_alloc(3, 2);
    // nl_mat_rand(m);
    // Mat m_t = nl_mat_alloc(m.cols, m.rows);
    // nl_mat_transpose(m_t, m);
    // nl_mat_print(m);
    // nl_mat_print(m_t);
    //
    // Mat m_col_0 = nl_mat_alloc(m_t.rows, 1);
    // nl_mat_get_col(m_col_0, m_t, 1);
    // nl_mat_print(m_col_0);
    //
    // nl_mat_free(m_col_0);
    // nl_mat_free(m_t);

    nl_mat_free(m);
    nl_mat_free(n);
    nl_mat_free(dst);

#else
    Arena arena = arena_new(8 * 1024);
    Mat a = nl_mat_alloc_with_arena(&arena, 5, 7);
    Mat b = nl_mat_alloc_with_arena(&arena, 2, 7);
    Mat arr[2] = {a, b};

    for (size_t r = 0; r < a.rows; ++r) {
        for (size_t c = 0; c < a.cols; ++c) {
            NL_MAT_AT(a, r, c) = c;
            NL_MAT_AT(b, r, c) = c;
        }
    }
    
    // nl_mat_print(a);

    // nl_mat_shuffle(a);
    // nl_mat_print(a);

    for (size_t i = 0; i < 2; ++i) nl_mat_print(arr[i]);
    nl_mat_shuffle_array(arr, 2);
    for (size_t i = 0; i < 2; ++i) nl_mat_print(arr[i]);

    arena_destroy(arena);

#endif

    return 0;
}
