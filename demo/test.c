#include <stdio.h>

#define NERUALIB_IMPLEMENTATION
#include "../neuralib.h"



int main(void)
{
    nl_rand_init(0, 0);
#if 1
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


    printf("----------\n");
    // Mat dot
    // nl_mat_one(n);
    // nl_mat_print(n);
    // nl_mat_dot(dst, m, n);
    // nl_mat_print(dst);

    printf("----------\n");
    // Mat multiply constant
    nl_mat_print(m);
    nl_mat_mult_c(m, m, 2.f);
    nl_mat_print(m);

    printf("----------\n");
    // Mat transpose
    nl_mat_free(m);
    m = nl_mat_alloc(3, 2);
    nl_mat_rand(m);
    Mat m_t = nl_mat_alloc(m.cols, m.rows);
    nl_mat_transpose(m_t, m);
    nl_mat_print(m);
    nl_mat_print(m_t);
    nl_mat_free(m_t);

    nl_mat_free(m);
    nl_mat_free(n);
    nl_mat_free(dst);
#else

#endif

    return 0;
}
