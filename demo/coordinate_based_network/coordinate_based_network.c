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

/* ------------------------------ */

void nl_model_save(const char *fname, NeuralNet model)
{
    FILE *fptr = fopen(fname, "w");
    if (!fptr) {
        fprintf(stderr, "Can NOT open file '%s' to save model\n", fname);
    }

    // Write count
    fprintf(fptr, "[Count]\n");
    fprintf(fptr, "%zu", model.count);
    fprintf(fptr, "\n\n");

    // Write layers
    fprintf(fptr, "[Layers]\n");
    for (size_t i = 0; i < model.count; ++i) {
        fprintf(fptr, "%zu ", model.layers[i]);
    }
    fprintf(fptr, "\n\n");

    // Write activations
    fprintf(fptr, "[Activations]\n");
    for (size_t i = 0; i < model.count - 1; ++i) {
        fprintf(fptr, "%u ", model.acts[i]);
    }
    fprintf(fptr, "\n\n");

    // Write cost function type
    fprintf(fptr, "[Cost_func]\n");
    fprintf(fptr, "%u", model.ct);
    fprintf(fptr, "\n\n");

    // Write weights
    fprintf(fptr, "[Weights]\n");
    for (size_t i = 0; i < model.count - 1; ++i) {
        fprintf(fptr, "{\n");
        for (size_t r = 0; r < model.ws[i].rows; ++r) {
            fprintf(fptr, "  ");
            for (size_t c = 0; c < model.ws[i].cols; ++c) {
                fprintf(fptr, "%f ", NL_MAT_AT(model.ws[i], r, c));
            }
            fprintf(fptr, "\n");
        }
        fprintf(fptr, "}\n");
    }
    fprintf(fptr, "\n");

    // Write biases
    fprintf(fptr, "[Biases]\n");
    for (size_t i = 0; i < model.count - 1; ++i) {
        fprintf(fptr, "{\n");
        for (size_t r = 0; r < model.bs[i].rows; ++r) {
            fprintf(fptr, "  ");
            for (size_t c = 0; c < model.bs[i].cols; ++c) {
                fprintf(fptr, "%f ", NL_MAT_AT(model.bs[i], r, c));
            }
            fprintf(fptr, "\n");
        }
        fprintf(fptr, "}\n");
    }

    fclose(fptr);
}

void nl_model_load(const char *fname, NeuralNet *model)
{
    FILE *fptr = fopen(fname, "r");
    if (!fptr) {
        fprintf(stderr, "Can NOT open file '%s' to read\n", fname);
    }

    char buf[512];
    // Count
    fscanf(fptr, "%s", buf);
    fscanf(fptr, "%zu", &(model->count));
    // printf("%zu\n", model->count);

    // Alloc memeroy
    model->layers = NL_MALLOC(sizeof(size_t) * model->count);
    model->acts = NL_MALLOC(sizeof(size_t) * (model->count - 1));
    model->ws = NL_MALLOC(sizeof(Mat) * (model->count - 1));
    model->bs = NL_MALLOC(sizeof(Mat) * (model->count - 1));

    // Layers
    fscanf(fptr, "%s", buf);
    for (size_t i = 0; i < model->count; ++i) {
        fscanf(fptr, "%zu ", &(model->layers[i]));
        printf("%zu ", model->layers[i]);
    }
    printf("\n");

    // Activations
    fscanf(fptr, "%s", buf);
    for (size_t i = 0; i < model->count - 1; ++i) {
        fscanf(fptr, "%u ", &(model->acts[i]));
        printf("%u ", model->acts[i]);
    }
    printf("\n");

    // Cost func
    fscanf(fptr, "%s", buf);
    fscanf(fptr, "%u", &(model->ct));
    printf("%s\n", buf);
    printf("%u\n", model->ct);
    printf("cost ok\n");

    // Weights
    fscanf(fptr, "%s", buf);
    for (size_t i = 0; i < model->count - 1; ++i) {
        fscanf(fptr, "%s", buf);
        model->ws[i] = nl_mat_alloc(model->layers[i + 1], model->layers[i]);
        for (size_t r = 0; r < model->ws[i].rows; ++r) {
            for (size_t c = 0; c < model->ws[i].cols; ++c) {
                fscanf(fptr, "%f ", &NL_MAT_AT(model->ws[i], r, c));
            }
        }
        fscanf(fptr, "%s", buf);
    }
    fscanf(fptr, "%s", buf);
    printf("weight ok\n");

    // Biases
    fscanf(fptr, "%s", buf);
    for (size_t i = 0; i < model->count - 1; ++i) {
        fscanf(fptr, "%s", buf);
        model->bs[i] = nl_mat_alloc(model->layers[i + 1], 1);
        for (size_t r = 0; r < model->bs[i].rows; ++r) {
            for (size_t c = 0; c < model->bs[i].cols; ++c) {
                fscanf(fptr, "%f ", &NL_MAT_AT(model->bs[i], r, c));
            }
        }
        fscanf(fptr, "%s", buf);
    }
    fscanf(fptr, "%s", buf);
    printf("bias ok\n");

    fclose(fptr);
}

void nl_model_load_with_arena(Arena *arena, const char *fname, NeuralNet *model)
{
    FILE *fptr = fopen(fname, "r");
    if (!fptr) {
        fprintf(stderr, "Can NOT open file '%s' to read\n", fname);
    }

    char buf[512];
    // Count
    fscanf(fptr, "%s", buf);
    fscanf(fptr, "%zu", &(model->count));

    // Alloc memeroy
    // model->layers = NL_MALLOC(sizeof(size_t) * model->count);
    model->layers = arena_alloc(arena, sizeof(size_t) * model->count);
    // model->acts = NL_MALLOC(sizeof(size_t) * (model->count - 1));
    model->acts = arena_alloc(arena, sizeof(size_t) * (model->count - 1));
    // model->ws = NL_MALLOC(sizeof(Mat) * (model->count - 1));
    model->ws = arena_alloc(arena, sizeof(Mat) * (model->count - 1));
    model->bs = arena_alloc(arena, sizeof(Mat) * (model->count - 1));

    // Layers
    fscanf(fptr, "%s", buf);
    for (size_t i = 0; i < model->count; ++i) {
        fscanf(fptr, "%zu ", &(model->layers[i]));
        printf("%zu ", model->layers[i]);
    }
    printf("\n");

    // Activations
    fscanf(fptr, "%s", buf);
    for (size_t i = 0; i < model->count - 1; ++i) {
        fscanf(fptr, "%u ", &(model->acts[i]));
        printf("%u ", model->acts[i]);
    }
    printf("\n");

    // Cost func
    fscanf(fptr, "%s", buf);
    fscanf(fptr, "%u", &(model->ct));
    printf("%s\n", buf);
    printf("%u\n", model->ct);
    printf("cost ok\n");

    // Weights
    fscanf(fptr, "%s", buf);
    for (size_t i = 0; i < model->count - 1; ++i) {
        fscanf(fptr, "%s", buf);
        model->ws[i] = nl_mat_alloc_with_arena(arena, model->layers[i + 1], model->layers[i]);
        for (size_t r = 0; r < model->ws[i].rows; ++r) {
            for (size_t c = 0; c < model->ws[i].cols; ++c) {
                fscanf(fptr, "%f ", &NL_MAT_AT(model->ws[i], r, c));
            }
        }
        fscanf(fptr, "%s", buf);
    }
    fscanf(fptr, "%s", buf);
    printf("weight ok\n");

    // Biases
    fscanf(fptr, "%s", buf);
    for (size_t i = 0; i < model->count - 1; ++i) {
        fscanf(fptr, "%s", buf);
        model->bs[i] = nl_mat_alloc_with_arena(arena, model->layers[i + 1], 1);
        for (size_t r = 0; r < model->bs[i].rows; ++r) {
            for (size_t c = 0; c < model->bs[i].cols; ++c) {
                fscanf(fptr, "%f ", &NL_MAT_AT(model->bs[i], r, c));
            }
        }
        fscanf(fptr, "%s", buf);
    }
    fscanf(fptr, "%s", buf);
    printf("bias ok\n");

    fclose(fptr);
}

void nl_model_summary(NeuralNet model, FILE *fd)
{
    fprintf(fd, "--------------------\n");
    fprintf(fd, "|   Model Summary  |\n");
    fprintf(fd, "--------------------\n");

    fprintf(fd, "Input layer: %zu\n", model.layers[0]);
    fprintf(fd, "--------------------\n");

    fprintf(fd, "Hidden layers:\n");
    char *act;
    size_t padding = 2;
    for (size_t i = 1; i < model.count - 1; ++i) {
        switch (model.acts[i-1]) {
            case SIGMOID:
                act = "Sigmoid";
                break;
            case RELU:
                act = "Relu";
                break;
        }
        fprintf(fd, "%*c%zu, %s\n", (int)padding, ' ', model.layers[i], act);
    }
    fprintf(fd, "--------------------\n");

    char *ct;
    switch (model.ct) {
        case MSE:
            ct = "Mean square error";
            break;
    }
    fprintf(fd, "Output layer: %zu, %s\n", model.layers[model.count - 1], ct);
}

/* ------------------------------ */

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
    size_t epochs = 1000 * 1;
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

    // nl_model_save("model.ml", model);

    NeuralNet new_model;
    // nl_model_load("model.ml", &new_model);

    // nl_model_load_with_arena(&arena, "model.ml", &new_model);

    // nl_model_save("test_model_read.ml", new_model);
    // nl_model_summary(model, stdout);

    // nl_model_free(new_model);

    arena_destroy(arena);
    return 0;
}
