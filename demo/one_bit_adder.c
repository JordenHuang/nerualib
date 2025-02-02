#include <stdio.h>

#define NERUALIB_IMPLEMENTATION
#include "../neuralib.h"

#define TRAIN_SIZE (sizeof(train)/sizeof(train[0]))

// 1-bit adder
float train[][5] = {
//    a  b  carry_in   sum carry_out
    { 0, 0,        0,    0,        0 },
    { 0, 1,        0,    1,        0 },
    { 1, 0,        0,    1,        0 },
    { 1, 1,        0,    0,        1 },
};

// void gen_train_data(void)
// {
//     printf("float train[][%d] = {\n", 5);
//     printf("//   ( a ) + ( b ) = (c)\n");
//     for (size_t i=0; i < 2; ++i) {
//         for (size_t j=0; j < 2; ++j) {
//             for (size_t k=0; k < 2; ++k) {
//                 for (size_t l=0; l < 2; ++l) {
//                     printf("    { %zu, %zu,  %zu, %zu,    %zu },\n",
//                            i, j, k, l,
//                            i*2+j + k*2+l
//                            );
//                 }
//             }
//         }
//     }
//     printf("};\n");
// }


int main(void)
{
    NeuralNet model;
    size_t layers[] = {3, 4, 2};
    nl_define_layers(&model, NL_ARRAY_LEN(layers), layers);

    Arena arena = arena_new(1024 * 1024);
    nl_rand_init(0, 0);

    // Train
    size_t epoch = 5000 * 100;
    float lr = 5e-3;
    Mat x = nl_mat_alloc_with_arena(&arena, 3, 1);
    Mat y = nl_mat_alloc_with_arena(&arena, 2, 1);
    for (size_t e = 0; e < epoch; ++e) {
        for (size_t i = 0; i < TRAIN_SIZE; ++i) {
            for (size_t a = 0; a < x.rows; ++a) {
                x.items[a] = train[i][a];
            }
            for (size_t a = 0; a < y.rows; ++a) {
                y.items[a] = train[i][a + x.rows];
            }
            // nl_mat_print(x);
            // nl_mat_print(y);
            nl_model_train(model, x, y, lr, SIGMOID, MSE);
        }
    }

    // Predict
    Mat px = nl_mat_alloc_with_arena(&arena, 3, 1);
    Mat py = nl_mat_alloc_with_arena(&arena, 2, 1);
    for (size_t i = 0; i < TRAIN_SIZE; ++i) {
        for (size_t a = 0; a < px.rows; ++a) {
            px.items[a] = train[i][a];
        }
        for (size_t a = 0; a < y.rows; ++a) {
            py.items[a] = train[i][a + x.rows];
        }
        nl_model_predict(model, px, py, SIGMOID);
        // printf("sum   : %f, cout: %f,\n", py.items[0], py.items[1]);
        // printf("expect: %f,       %f\n", train[i][px.rows], train[i][px.rows + 1]);

        py.items[0] = (py.items[0] < 0.5) ? 0.f: 1.f;
        py.items[1] = (py.items[1] < 0.5) ? 0.f: 1.f;
        train[i][px.rows]     = (train[i][px.rows] < 0.5)     ? 0.f: 1.f;
        train[i][px.rows + 1] = (train[i][px.rows + 1] < 0.5) ? 0.f: 1.f;

        printf("%.2f + %.2f\n", px.items[0], px.items[1]);
        printf("sum   : %.2f, cout: %.2f,\n",
               py.items[0],
               py.items[1]);
        printf("expect: %.2f,       %.2f",
               train[i][px.rows],
               train[i][px.rows + 1]);
        if (py.items[0] == train[i][px.rows] && py.items[1] == train[i][px.rows + 1]) {
            printf(" (SAME)\n");
        } else {
            printf("\n");
        }
        printf("\n");
    }

    printf("===== Final\n");
    for (size_t i = 0; i < NL_ARRAY_LEN(layers) - 1; ++i) {
        nl_mat_print(model.ws[i]);
        nl_mat_print(model.bs[i]);
    }

    nl_model_free(model);
    arena_destroy(arena);
}
