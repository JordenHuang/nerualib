#include <stdio.h>

#define NERUALIB_IMPLEMENTATION
#include "neuralib.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(void)
{
    nl_rand_init(false, 0);
    NL_SET_ARENA_CAPACITY(64 * 1024 * 1024);

    Arena arena = arena_new(64 * 1024 * 1024);
    NeuralNet model;

    const char model_path[] = "./image_upscaler/upscaler.model";
    nl_model_load_with_arena(&arena, model_path, &model);
    printf("Model loaded\n");

    nl_model_summary(model, stdout);

    // nl_model_accuracy_score();


#if 1
    printf("===== Scale up\n");
    int w = 28 * 10;
    int h = w;
    int n = 1;
    size_t out_w = w;
    size_t out_h = h;
    float h_fact;
    float w_fact;

    Mat test_x = nl_mat_alloc_with_arena(&arena, 8, 1);
    Mat predicted = nl_mat_alloc_with_arena(&arena, n, 1);
    unsigned char *img_out_data = arena_alloc(&arena, sizeof(unsigned char) * out_w * out_h * n);

    for (size_t i = 0; i < out_h; ++i) {
        h_fact = (float)i / (float)(out_h-1);
        for (size_t j = 0; j < out_w; ++j) {
            w_fact = (float)j / (float)(out_w-1);
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

            for (size_t k = 0; k < (size_t)n; ++k) {
                img_out_data[i*out_w*n + j*n + k] = (unsigned char)(NL_MAT_AT(predicted, k, 0) * 255);
            }
        }
    }
    stbi_write_png("out_2.png", out_w, out_h, n, img_out_data, out_w*n);
    printf("Image generated\n");
#endif

    arena_destroy(arena);
    return 0;
}
