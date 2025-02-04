#include <stdio.h>

#define NERUALIB_IMPLEMENTATION
#include "neuralib.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMANIP_IMPLEMENTATION
#include "imanip.h"

#define MIN(a, b) (a < b ? a : b)

// https://medium.com/@chathuragunasekera/image-resampling-algorithms-for-pixel-manipulation-bee65dda1488
void resize_image(Img out, Img in)
{
    float w_factor = (float)in.w / (float)out.w;
    float h_factor = (float)in.h / (float)out.h;
    float src_x, src_y;
    for (int h = 0; h < out.h; ++h) {
        for (int w = 0; w < out.w; ++w) {
            src_x = w * w_factor;
            src_y = h * h_factor;
            int x1 = (int)src_x;
            int y1 = (int)src_y;
            int x2 = MIN(x1+1, in.w-1);
            int y2 = MIN(y1+1, in.h-1);

            // Compute the interpolation coefficients
            int alpha = src_x - x1;
            int beta = src_y - y1;

            // Perform bilinear interpolation
            for (int c=0; c < out.channel; ++c) {
                out.data[w + h*out.w + c] =
                    (1 - alpha) * (1 - beta) * in.data[x1 + y1*out.w + c] +
                    alpha * (1 - beta) * in.data[x2 + y1*out.w + c] +
                    + (1 - alpha) * beta * in.data[x1 + y2*out.w + c]
                    + alpha * beta * in.data[x2 + y2*out.w + c];
            }
        }
    }
}

void img_flatten(Mat dst, Img img)
{
    for (size_t i = 0; i < dst.rows; ++i) {
        dst.items[i] = img.data[i];
    }
}

void mat_to_img(Img dst, Mat m)
{
    for (int i = 0; i < dst.w*dst.h; ++i) {
        dst.data[i] = m.items[i];
    }
}

int main(void)
{
    Img img;
    printf("===== Load image\n");
    // const char filename[] = "image_samples/cup.jpg";
    const char filename[] = "image_samples/baboon.png";
    img.data = stbi_load(filename, &img.w, &img.h, &img.channel, 0);
    if (img.data == NULL) {
        fprintf(stderr, "[ERROR] Can NOT load image '%s'\n", filename);
        exit(1);
    }
    printf("width: %d, height: %d, channel: %d\n", img.w, img.h, img.channel);

    printf("===== To grayscale\n");
    // training x
    Img gray_img;
    iman_img_new(&gray_img, img.w, img.h, 1);
    iman_grayscale(gray_img, img);

    printf("===== Resizing\n");
    // training y
    Img smaller_gray_img;
    int f = 8;
    iman_img_new(&smaller_gray_img, gray_img.w / f, gray_img.h / f, gray_img.channel);
    // iman_gaussian_blur(smaller_gray_img, gray_img, 5, 5, 1.7, 1.7);
    resize_image(smaller_gray_img, gray_img);

    const char gray_filename[] = "gray.png";
    const char smaller_filename[] = "smaller.png";
    stbi_write_png(gray_filename, gray_img.w, gray_img.h, gray_img.channel, gray_img.data, gray_img.w*gray_img.channel);
    stbi_write_png(smaller_filename, smaller_gray_img.w, smaller_gray_img.h, smaller_gray_img.channel, smaller_gray_img.data, smaller_gray_img.w*smaller_gray_img.channel);

    printf("===== Set some parameters\n");
    NL_SET_ARENA_CAPACITY(512 * 1024 * 1024);
    nl_rand_init(0, 0);
    NL_PRINT_COST_EVERY_N_EPOCHS(20);

    printf("===== Prepare training data\n");
    // Prepare training data
    Arena arena = arena_new(640 * 1024 * 1024);
    Mat x = nl_mat_alloc_with_arena(&arena, smaller_gray_img.w * smaller_gray_img.h, 1);
    Mat y = nl_mat_alloc_with_arena(&arena, gray_img.w * gray_img.h, 1);
    arena_info(arena);
    printf("----- Flatten\n");
    img_flatten(x, smaller_gray_img);
    img_flatten(y, gray_img);
    printf("----- Normalize\n");
    nl_mat_mult_c(x, x, 1.f/255.f);
    nl_mat_mult_c(y, y, 1.f/255.f);

    printf("===== Define model layer components\n");
    NeuralNet model;
    size_t layers[] = {
        smaller_gray_img.w * smaller_gray_img.h,
        1,
        // 16,
        gray_img.w * gray_img.h
    };
    // Activation_type acts[] = {RELU, RELU, RELU, SIGMOID};
    Activation_type acts[] = {SIGMOID, SIGMOID};
    printf("===== Define model layers\n");
    nl_define_layers_with_arena(&arena, &model, NL_ARRAY_LEN(layers), layers, acts, MSE);

    float lr = 5e-2;
    size_t epochs = 500;
    printf("===== Train\n");
    nl_model_train(model, x, y, lr, epochs, 1, false);

    printf("===== Predict\n");
    Mat predicted = nl_mat_alloc_with_arena(&arena, gray_img.w * gray_img.h, 1);
    // nl_mat_rand(x);
    nl_model_predict(model, x, predicted);

    Img out;
    nl_mat_mult_c(predicted, predicted, 255.f);
    iman_img_new(&out, gray_img.w, gray_img.h, gray_img.channel);
    // // No matter what the input is, it will always generate a image that looks like the training image
    // mat_to_img(out, x);
    // stbi_write_png("x.png", out.w, out.h, out.channel, out.data, out.w*out.channel);
    mat_to_img(out, predicted);

    const char out_filename[] = "out.png";
    stbi_write_png(out_filename, out.w, out.h, out.channel, out.data, out.w*out.channel);
    printf("Check '%s'\n\n", out_filename);
    printf("What this network do is\n");
    printf("No matter what the input is, it will always generate a image that looks like the training image\n");

    arena_destroy(arena);
    stbi_image_free(img.data);
    iman_img_free(&gray_img);
    iman_img_free(&smaller_gray_img);
    iman_img_free(&out);
    return 0;
}
