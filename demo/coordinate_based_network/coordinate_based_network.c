#include <stdio.h>

#define NERUALIB_IMPLEMENTATION
#include "neuralib.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// #define IMANIP_IMPLEMENTATION
// #include "imanip.h"

// void img_flatten(Mat dst, Img img)
// {
//     for (size_t i = 0; i < dst.rows; ++i) {
//         dst.items[i] = img.data[i];
//     }
// }
//
// void mat_to_img(Img dst, Mat m)
// {
//     for (int i = 0; i < dst.w*dst.h; ++i) {
//         dst.data[i] = m.items[i];
//     }
// }

void arg_shift(int *argc, char **argv[])
{
    if (*argc > 0) {
        (*argc) -= 1;
        (*argv) += 1;
    } else {
        fprintf(stderr, "[ERROR] No more arguments\n");
        exit(1);
    }
}

void display_usage(void)
{
    printf("Usage: coordinate_based_network.out [image path]\n");
}

int main(int argc, char *argv[])
{
    // char *program_name = argv[0];
    arg_shift(&argc, &argv);
    if (argc == 0) {
        fprintf(stderr, "No input is provided\n");
        display_usage();
    }

    char *img_in_path = argv[0];
    // arg_shift(&argc, &argv);
    printf("Input: %s\n", img_in_path);
    int w, h, n;
    unsigned char *img_in_data = stbi_load(img_in_path, &w, &h, &n, 0);
    if (img_in_data == NULL) {
        fprintf(stderr, "[ERROR] Can NOT load image '%s'\n", img_in_path);
        exit(1);
    }
    printf("  width: %d, height: %d, channel: %d\n", w, h, n);

    printf("===== Set some parameters\n");
    NL_SET_ARENA_CAPACITY(64 * 1024 * 1024);
    nl_rand_init(false, 0);
    NL_PRINT_COST_EVERY_N_EPOCHS(100);

    printf("===== Prepare training data\n");
    Arena arena = arena_new(64 * 1024 * 1024);
    Mat xs = nl_mat_alloc_with_arena(&arena, 8, w * h);
    Mat ys = nl_mat_alloc_with_arena(&arena, 1, w * h);

    float h_fact, w_fact;
    for (int i = 0; i < h; ++i) {
        h_fact = (float)i / (float)(h-1);
        for (int j = 0; j < w; ++j) {
            w_fact = (float)j / (float)(w-1);
            // NL_MAT_AT(xs, 0, j + i*w) = w_fact;
            // NL_MAT_AT(xs, 1, j + i*w) = h_fact;
            NL_MAT_AT(ys, 0, j + i*w) = img_in_data[i*w + j] / 255.f;

            NL_MAT_AT(xs, 0, j + i*w) = sin(1 * M_PI * w_fact);
            NL_MAT_AT(xs, 1, j + i*w) = cos(1 * M_PI * w_fact);
            NL_MAT_AT(xs, 2, j + i*w) = sin(2 * M_PI * w_fact);
            NL_MAT_AT(xs, 3, j + i*w) = cos(2 * M_PI * w_fact);
            // Repeat for y
            NL_MAT_AT(xs, 4, j + i*w) = sin(1 * M_PI * h_fact);
            NL_MAT_AT(xs, 5, j + i*w) = cos(1 * M_PI * h_fact);
            NL_MAT_AT(xs, 6, j + i*w) = sin(2 * M_PI * h_fact);
            NL_MAT_AT(xs, 7, j + i*w) = cos(2 * M_PI * h_fact);
        }
    }

    printf("===== Define model layer components\n");
    NeuralNet model;
    // size_t layers[] = {8, 4, 7, 4, 1};
    // Activation_type acts[] = {SIGMOID, SIGMOID, SIGMOID, SIGMOID};
    size_t layers[] = {8, 7, 4, 1};
    Activation_type acts[] = {SIGMOID, SIGMOID, SIGMOID};

    printf("===== Define model layers\n");
    nl_define_layers_with_arena(&arena, &model, NL_ARRAY_LEN(layers), layers, acts, MSE);

    printf("===== Train\n");
    float lr = 7e-2;
    size_t epochs = 1000 * 8;
    size_t batch_size = 4;
    nl_model_train(model, xs, ys, lr, epochs, batch_size, false);

    // printf("===== Predict\n");
    printf("===== Scale up\n");
    Mat test_x = nl_mat_alloc_with_arena(&arena, 8, 1);
    Mat predicted = nl_mat_alloc_with_arena(&arena, 1, 1);
    size_t out_w = w * 25;
    size_t out_h = h * 25;
    unsigned char *img_out_data = arena_alloc(&arena, sizeof(unsigned char) * out_w * out_h * n);
    for (size_t i = 0; i < out_h; ++i) {
        h_fact = (float)i / (float)(out_h-1);
        for (size_t j = 0; j < out_w; ++j) {
            w_fact = (float)j / (float)(out_w-1);
            // NL_MAT_AT(test_x, 0, 0) = w_fact;
            // NL_MAT_AT(test_x, 1, 0) = h_fact;

            NL_MAT_AT(test_x, 0, 0) = sin(1 * M_PI * w_fact);
            NL_MAT_AT(test_x, 1, 0) = cos(1 * M_PI * w_fact);
            NL_MAT_AT(test_x, 2, 0) = sin(2 * M_PI * w_fact);
            NL_MAT_AT(test_x, 3, 0) = cos(2 * M_PI * w_fact);
            // Repeat for y
            NL_MAT_AT(test_x, 4, 0) = sin(1 * M_PI * h_fact);
            NL_MAT_AT(test_x, 5, 0) = cos(1 * M_PI * h_fact);
            NL_MAT_AT(test_x, 6, 0) = sin(2 * M_PI * h_fact);
            NL_MAT_AT(test_x, 7, 0) = cos(2 * M_PI * h_fact);

            nl_model_predict(model, test_x, predicted);

            img_out_data[i*out_w + j] = (unsigned char)(NL_MAT_AT(predicted, 0, 0) * 255);
        }
    }
    stbi_write_png("out.png", out_w, out_h, n, img_out_data, out_w*n);

    arena_destroy(arena);
    return 0;
}
