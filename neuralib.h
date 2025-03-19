/*
References:
- https://www.3blue1brown.com/lessons/backpropagation-calculus
- http://neuralnetworksanddeeplearning.com/chap2.html
- https://datasciocean.tech/deep-learning-core-concept/backpropagation-explain/
*/

#ifndef _NERUALIB_H_
#define _NERUALIB_H_

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#define NL_DEBUGGING 1
#if (NL_DEBUGGING == 0)
  #define NDEBUG
#endif

#ifndef NL_MALLOC
#define NL_MALLOC(sz) malloc(sz)
#endif // NL_MALLOC

#ifndef NL_FREE
#define NL_FREE(p) free(p)
#endif // NL_FREE

#ifndef NL_ASSERT
#define NL_ASSERT(v) assert(v)
#endif // NL_ASSERT

#ifndef NL_FSCANF
#define NL_FSCANF(stream, fmt, ...) if (fscanf(stream, fmt, __VA_ARGS__)) {}
#endif // NL_FSCANF

#define NL_ARRAY_LEN(arr) (sizeof(arr) / sizeof(arr[0]))
#define NL_MAT_INDEX(columns, r, c) ((r)*(columns) + (c))
#define NL_MAT_AT_INDEX(m, idx) (m.items[idx])
#define NL_MAT_AT(m, r, c) (m.items[NL_MAT_INDEX(m.cols, r, c)])

// Global variable
size_t _nl_n_epochs = 1;
#define NL_PRINT_COST_EVERY_N_EPOCHS(n) _nl_n_epochs = n

size_t nl_arena_capacity = 64 * 1024 * 1024;
#define NL_SET_ARENA_CAPACITY(cap) nl_arena_capacity = cap

typedef struct {
    size_t capacity;
    size_t size;
    uintptr_t *begin;
} Arena;

Arena arena_new(size_t capacity);
void arena_destroy(Arena a);
void arena_info(Arena a);

void *arena_alloc(Arena *a, size_t sz);
void arena_reset(Arena *a);


typedef enum {
    SIGMOID,
    RELU,
} Activation_type;

typedef enum {
    MSE,
} Cost_type;

float sigmoidf(float z);
float sigmoidf_prime(float z);
float relu(float z);
float relu_prime(float z);

float mse(float a, float y);
float mse_prime(float a, float y);


typedef struct {
    size_t rows;
    size_t cols;
    float *items;
} Mat;

void nl_rand_init(bool use_seed, size_t seed);
float nl_rand_float(void);
Mat nl_mat_alloc(size_t row, size_t col);
Mat nl_mat_alloc_with_arena(Arena *arena, size_t row, size_t col);
void nl_mat_zero(Mat m);
void nl_mat_one(Mat m);
void nl_mat_rand(Mat m);
void _nl_mat_print(Mat m, const char *name, size_t padding);
#define nl_mat_print(m) _nl_mat_print(m, #m, 0);
void nl_mat_get_col(Mat dst, Mat m, size_t col);
void nl_mat_add(Mat dst, Mat m1, Mat m2);
void nl_mat_mult_c(Mat dst, Mat m, float c);
void nl_mat_mult(Mat dst, Mat m1, Mat m2);
void nl_mat_dot(Mat dst, Mat a, Mat b);
void nl_mat_transpose(Mat dst, Mat m);
void nl_mat_copy(Mat dst, Mat m);
void nl_mat_free(Mat m);

void nl_mat_act(Mat dst, Mat m, Activation_type act);
void nl_mat_act_prime(Mat dst, Mat m, Activation_type act);
float nl_mat_cost(Mat dst, Mat m, Mat ys, Cost_type ct);
void nl_mat_cost_prime(Mat dst, Mat m, Mat ys, Cost_type ct);


typedef struct {
    size_t count;
    size_t *layers; // Input, hidden, output layers, input layer doesn't have w and b
    Activation_type *acts; // Activation function type for hidden layers and output layer, length should be (count-1)
    Cost_type ct; // Cost function type
    Mat *ws; // Array of matrices, a row a neuron
    Mat *bs; // Array of column vectors
} NeuralNet;

void nl_define_layers(NeuralNet *model, size_t count, size_t *layers, Activation_type *acts, Cost_type ct);
void nl_define_layers_with_arena(Arena *arena, NeuralNet *model, size_t count, size_t *layers, Activation_type *acts, Cost_type ct);
void nl_model_summary(NeuralNet model, FILE *fd);
void nl_model_train(NeuralNet model, Mat xs, Mat ys, float lr, size_t epochs, size_t batch_size, bool shuffle);
void nl_model_feed_forward(NeuralNet model, Mat *zsa, Mat *asa);
void nl_model_backprop(NeuralNet model, Mat ys, Mat *zsa, Mat *asa, Mat *delta_ws, Mat *delta_bs);
void nl_model_predict(NeuralNet model, Mat ins, Mat outs);
float nl_model_accuracy_score(Mat y_true, Mat y_predict);
void nl_model_free(NeuralNet model);
void nl_model_save(const char *fname, NeuralNet model);
void nl_model_load(const char *fname, NeuralNet *model);
void nl_model_load_with_arena(Arena *arena, const char *fname, NeuralNet *model);

#endif // _NERUALIB_H_

#ifdef NERUALIB_IMPLEMENTATION

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

Arena arena_new(size_t bytes)
{
    Arena a;
    a.capacity = bytes;
    a.size = 0;
    a.begin = NL_MALLOC(bytes);
    return a;
}

void arena_destroy(Arena a)
{
    NL_FREE(a.begin);
}

void arena_info(Arena a)
{
    printf("capacity: %zu\n", a.capacity);
    printf("size: %zu\n", a.size);
    printf("begin: %p\n", a.begin);
}

void *arena_alloc(Arena *a, size_t sz)
{
    if ((a->size + sz) <= a->capacity) {
        size_t offset = a->size;
        a->size += sz;
        return a->begin + offset;
    } else {
        fprintf(stderr, "[ERROR] Not enough capacity for this region\n");
        exit(1);
        return NULL;
    }
}

void arena_reset(Arena *a)
{
    a->size = 0;
}


float sigmoidf(float z)
{
    return 1.f / (1.f + expf(-z));
}

float sigmoidf_prime(float z)
{
    return sigmoidf(z) * (1.f - sigmoidf(z));
}

float relu(float z)
{
    return (z > 0.f) ? z : 0.f;
}

float relu_prime(float z)
{
    return (z > 0.f) ? 1.f : 0.f;
}

float mse(float a, float y)
{
    return (a - y) * (a - y);
}

float mse_prime(float a, float y)
{
    return 2.f * (a - y);
}


void nl_rand_init(bool use_seed, size_t seed)
{
    if (use_seed) srand(seed);
    else srand(time(NULL));
}

float nl_rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}

Mat nl_mat_alloc(size_t row, size_t col)
{
    Mat m;
    m.rows = row;
    m.cols = col;
    m.items = NL_MALLOC(sizeof(float) * row * col);
    return m;
}

Mat nl_mat_alloc_with_arena(Arena *arena, size_t row, size_t col)
{
    Mat m;
    m.rows = row;
    m.cols = col;
    m.items = arena_alloc(arena, sizeof(float) * row * col);
    return m;
}

void nl_mat_zero(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
           NL_MAT_AT(m, i, j) = 0.f;
        }
    }
}

void nl_mat_one(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
           NL_MAT_AT(m, i, j) = 1.f;
        }
    }
}

void nl_mat_rand(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
           // NL_MAT_AT(m, i, j) = nl_rand_float();
           // NL_MAT_AT(m, i, j) = nl_rand_float() * (1.f - (-1.f)) + (-1.f);
           NL_MAT_AT(m, i, j) = nl_rand_float() * (2.f) - 1.f;
        }
    }
}

void _nl_mat_print(Mat m, const char *name, size_t padding)
{
    printf("%*sshape = (%zu, %zu)\n", (int)padding, "", m.rows, m.cols);
    printf("%*s%s = [\n", (int)padding, "", name);
    for (size_t i = 0; i < m.rows; ++i) {
        printf("%*s  ", (int)padding, "");
        for (size_t j = 0; j < m.cols; ++j) {
            printf("%f, ", NL_MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}

void nl_mat_get_col(Mat dst, Mat m, size_t col)
{
    NL_ASSERT(dst.rows == m.rows);
    NL_ASSERT(dst.cols == 1);
    for (size_t i = 0; i < dst.rows; ++i) {
        dst.items[i] = NL_MAT_AT(m, i, col);
    }
}

void nl_mat_add(Mat dst, Mat m1, Mat m2)
{
    NL_ASSERT(m1.rows == m2.rows);
    NL_ASSERT(m1.cols == m2.cols);
    NL_ASSERT(dst.rows == m1.rows);
    NL_ASSERT(dst.cols == m1.cols);
    for (size_t i = 0; i < dst.rows * dst.cols; ++i) {
        dst.items[i] = m1.items[i] + m2.items[i];
    }
}

void nl_mat_mult_c(Mat dst, Mat m, float c)
{
    NL_ASSERT(dst.rows == m.rows);
    NL_ASSERT(dst.cols == m.cols);
    for (size_t i = 0; i < dst.rows * dst.cols; ++i) {
        dst.items[i] = m.items[i] * c;
    }
}

void nl_mat_mult(Mat dst, Mat m1, Mat m2)
{
    NL_ASSERT(m1.rows == m2.rows);
    NL_ASSERT(m1.cols == m2.cols);
    NL_ASSERT(dst.rows == m1.rows);
    NL_ASSERT(dst.cols == m1.cols);
    for (size_t i = 0; i < dst.rows * dst.cols; ++i) {
        dst.items[i] = m1.items[i] * m2.items[i];
    }
}

void nl_mat_dot(Mat dst, Mat m1, Mat m2)
{
#if 0
    NL_ASSERT(m1.cols == m2.rows);
    NL_ASSERT(dst.rows == m1.rows);
    NL_ASSERT(dst.cols == m2.cols);
    for (size_t r = 0; r < dst.rows; ++r) {
        for (size_t c = 0; c < dst.cols; ++c) {
            size_t idx = NL_MAT_INDEX(dst.cols, r, c);
            NL_MAT_AT_INDEX(dst, idx) = 0.f;
            for (size_t cr = 0; cr < m1.cols; ++cr) {
                NL_MAT_AT_INDEX(dst, idx) += NL_MAT_AT(m1, r, cr) * NL_MAT_AT(m2, cr, c);
            }
        }
    }
#else
    Mat m2_trans = nl_mat_alloc(m2.cols, m2.rows);
    for (size_t i = 0; i < m2.rows * m2.cols; ++i) {
        m2_trans.items[i] = m2.items[i];
    }
    for (size_t r = 0; r < dst.rows; ++r) {
        for (size_t c = 0; c < dst.cols; ++c) {
            size_t idx = NL_MAT_INDEX(dst.cols, r, c);
            NL_MAT_AT_INDEX(dst, idx) = 0.f;
            for (size_t cr = 0; cr < m1.cols; ++cr) {
                NL_MAT_AT_INDEX(dst, idx) += NL_MAT_AT(m1, r, cr) * NL_MAT_AT(m2_trans, c, cr);
            }
        }
    }
    nl_mat_free(m2_trans);
#endif
}

void nl_mat_transpose(Mat dst, Mat m)
{
    NL_ASSERT(dst.rows == m.cols);
    NL_ASSERT(dst.cols == m.rows);
    for (size_t r = 0; r < m.rows; ++r) {
        for (size_t c = 0; c < m.cols; ++c) {
            NL_MAT_AT(dst, c, r) = NL_MAT_AT(m, r, c);
        }
    }
}

void nl_mat_copy(Mat dst, Mat m)
{
    NL_ASSERT(dst.rows == m.rows);
    NL_ASSERT(dst.cols == m.cols);
    for (size_t r = 0; r < dst.rows; ++r) {
        for (size_t c = 0; c < dst.cols; ++c) {
            NL_MAT_AT(dst, r, c) = NL_MAT_AT(m, r, c);
        }
    }
}

void nl_mat_act(Mat dst, Mat m, Activation_type act)
{
    NL_ASSERT(dst.rows == m.rows);
    NL_ASSERT(dst.cols == m.cols);
    float (*act_fn)(float);
    switch (act) {
        case RELU:
            act_fn = relu;
            break;
        case SIGMOID:
        default:
            act_fn = sigmoidf;
            break;
    }
    for (size_t r = 0; r < dst.rows; ++r) {
        for (size_t c = 0; c < dst.cols; ++c) {
            NL_MAT_AT(dst, r, c) = act_fn(NL_MAT_AT(m, r, c));
        }
    }
}

void nl_mat_act_prime(Mat dst, Mat m, Activation_type act)
{
    NL_ASSERT(dst.rows == m.rows);
    NL_ASSERT(dst.cols == m.cols);
    float (*act_fn_prime)(float);
    switch (act) {
        case RELU:
            act_fn_prime = relu_prime;
            break;
        case SIGMOID:
        default:
            act_fn_prime  = sigmoidf_prime;
            break;
    }
    for (size_t r = 0; r < dst.rows; ++r) {
        for (size_t c = 0; c < dst.cols; ++c) {
            NL_MAT_AT(dst, r, c) = act_fn_prime(NL_MAT_AT(m, r, c));
        }
    }
}

float nl_mat_cost(Mat dst, Mat m, Mat ys, Cost_type ct)
{
    NL_ASSERT(m.rows == ys.rows);
    NL_ASSERT(m.cols == ys.cols);
    NL_ASSERT(m.cols == 1);
    NL_ASSERT(dst.rows == m.rows);
    NL_ASSERT(dst.cols == m.cols);
    float (*loss_fn)(float, float);
    switch (ct) {
        case MSE:
        default:
            loss_fn = mse;
            break;
    }
    float cost = 0.f;
    for (size_t r = 0; r < m.rows; ++r) {
        NL_MAT_AT(dst, r, 0) = loss_fn(NL_MAT_AT(m, r, 0), NL_MAT_AT(ys, r, 0));
        cost += NL_MAT_AT(dst, r, 0);
    }
    return cost/(float)m.rows;
}

void nl_mat_cost_prime(Mat dst, Mat m, Mat ys, Cost_type ct)
{
    NL_ASSERT(m.rows == ys.rows);
    NL_ASSERT(m.cols == ys.cols);
    NL_ASSERT(m.cols == 1);
    NL_ASSERT(dst.rows == m.rows);
    NL_ASSERT(dst.cols == m.cols);
    float (*loss_fn_prime)(float, float);
    switch (ct) {
        case MSE:
        default:
            loss_fn_prime = mse_prime;
            break;
    }
    for (size_t r = 0; r < m.rows; ++r) {
        NL_MAT_AT(dst, r, 0) = loss_fn_prime(NL_MAT_AT(m, r, 0), NL_MAT_AT(ys, r, 0));
    }
}

void nl_mat_free(Mat m)
{
    NL_FREE(m.items);
}


void nl_define_layers(NeuralNet *model, size_t count, size_t *layers, Activation_type *acts, Cost_type ct)
{
    model->count = count;
    model->ct = ct;
    model->layers = NL_MALLOC(sizeof(size_t) * count);
    for (size_t i = 0; i < count; ++i) {
        model->layers[i] = layers[i];
    }
    model->acts = NL_MALLOC(sizeof(size_t) * (count - 1));
    model->ws = NL_MALLOC(sizeof(Mat) * (count - 1));
    model->bs = NL_MALLOC(sizeof(Mat) * (count - 1));
    for (size_t i = 1; i < count; ++i) {
        model->acts[i-1] = acts[i-1];
        model->ws[i-1] = nl_mat_alloc(layers[i], layers[i - 1]);
        model->bs[i-1] = nl_mat_alloc(layers[i], 1);
        nl_mat_rand(model->ws[i-1]);
        nl_mat_rand(model->bs[i-1]);
    }
}

void nl_define_layers_with_arena(Arena *arena, NeuralNet *model, size_t count, size_t *layers, Activation_type *acts, Cost_type ct)
{
    model->count = count;
    model->ct = ct;
    model->layers = arena_alloc(arena, sizeof(size_t) * count);
    for (size_t i = 0; i < count; ++i) {
        model->layers[i] = layers[i];
    }
    model->acts = arena_alloc(arena, sizeof(size_t) * (count - 1));
    model->ws = arena_alloc(arena, sizeof(Mat) * (count - 1));
    model->bs = arena_alloc(arena, sizeof(Mat) * (count - 1));
    for (size_t i = 1; i < count; ++i) {
        model->acts[i-1] = acts[i-1];
        model->ws[i-1] = nl_mat_alloc_with_arena(arena, layers[i], layers[i - 1]);
        model->bs[i-1] = nl_mat_alloc_with_arena(arena, layers[i], 1);
        nl_mat_rand(model->ws[i-1]);
        nl_mat_rand(model->bs[i-1]);
    }
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

// https://medium.com/%E5%AD%B8%E4%BB%A5%E5%BB%A3%E6%89%8D/%E5%84%AA%E5%8C%96%E6%BC%94%E7%AE%97-5a4213d08943
void nl_model_train(NeuralNet model, Mat xs, Mat ys, float lr, size_t epochs, size_t batch_size, bool shuffle)
{
    /*
     * - Set batch_size to 1 equals Stochastic Gradient Descent
     * - Set batch_size to 1 < n < xs.cols(=training data size) equals Mini-batch Gradient Descent
     * - Set batch_size to xs.cols(=training data size) equals Batch Gradient Descent
     */
    Arena arena = arena_new(nl_arena_capacity);

    // Alloc memoy for zs, array of column vectors
    Mat *zsa = arena_alloc(&arena, sizeof(Mat) * (model.count - 1));
    for (size_t i = 1; i < model.count; ++i) {
        zsa[i-1] = nl_mat_alloc_with_arena(&arena, model.layers[i], 1);
        nl_mat_zero(zsa[i-1]);
    }

    // Alloc memory for activations, array of column vectors
    Mat *asa = arena_alloc(&arena, sizeof(Mat) * model.count);
    for (size_t i = 0; i < model.count; ++i) {
        asa[i] = nl_mat_alloc_with_arena(&arena, model.layers[i], 1);
    }

    // Alloc memory for gradients to model.ws and model.bs
    Mat *delta_ws = arena_alloc(&arena, sizeof(Mat) * (model.count - 1));
    Mat *delta_bs = arena_alloc(&arena, sizeof(Mat) * (model.count - 1));
    // Alloc memory for total gradients to update to model.ws and model.bs
    Mat *nabla_ws = arena_alloc(&arena, sizeof(Mat) * (model.count - 1));
    Mat *nabla_bs = arena_alloc(&arena, sizeof(Mat) * (model.count - 1));
    for (size_t i = 1; i < model.count; ++i) {
        delta_ws[i-1] = nl_mat_alloc_with_arena(&arena, model.layers[i], model.layers[i - 1]);
        delta_bs[i-1] = nl_mat_alloc_with_arena(&arena, model.layers[i], 1);
        nabla_ws[i-1] = nl_mat_alloc_with_arena(&arena, model.layers[i], model.layers[i - 1]);
        nabla_bs[i-1] = nl_mat_alloc_with_arena(&arena, model.layers[i], 1);
            nl_mat_zero(nabla_ws[i-1]);
            nl_mat_zero(nabla_bs[i-1]);
    }

    Mat y = nl_mat_alloc_with_arena(&arena, ys.rows, 1);
    Mat losses = nl_mat_alloc_with_arena(&arena, asa[model.count-1].rows, asa[model.count-1].cols);
    float cost = 0.f;
    bool flag = false;

    // No shuffle
    NL_ASSERT(xs.cols % batch_size == 0);
    if (xs.cols % batch_size != 0) {
        fprintf(stderr, "xs.cols %% batch_size != 0\n");
        exit(1);
    }

    for (size_t e = 0; e < epochs; ++e) {
        if ((e+1) % _nl_n_epochs == 0) {
            cost = 0.f;
            flag = true;
        }

        for (size_t i = 0; i < xs.cols; i+=batch_size) {
            for (size_t b = 0; b < batch_size; ++b) {
                // Put training data x to asa[0]
                nl_mat_get_col(asa[0], xs, i + b);
                nl_mat_get_col(y, ys, i + b);

                // Forward pass
                nl_model_feed_forward(model, zsa, asa);
                    if (flag) cost += nl_mat_cost(losses, asa[model.count-1], y, model.ct);

                // Backward pass (backpropagaton)
                nl_model_backprop(model, y, zsa, asa, delta_ws, delta_bs);

                // Add up the nabla_ws and nabla_bs
                for (size_t l = 1; l < model.count; ++l) {
                    nl_mat_add(nabla_ws[l-1], nabla_ws[l-1], delta_ws[l-1]);
                    nl_mat_add(nabla_bs[l-1], nabla_bs[l-1], delta_bs[l-1]);
                }
            }

            // Update weights and bias
            for (size_t l = 1; l < model.count; ++l) {
                nl_mat_mult_c(nabla_ws[l-1], nabla_ws[l-1], (-lr) / batch_size);
                nl_mat_add(model.ws[l-1], model.ws[l-1], nabla_ws[l-1]);

                nl_mat_mult_c(nabla_bs[l-1], nabla_bs[l-1], (-lr) / batch_size);
                nl_mat_add(model.bs[l-1], model.bs[l-1], nabla_bs[l-1]);

                nl_mat_zero(nabla_ws[l-1]);
                nl_mat_zero(nabla_bs[l-1]);
            }
        }

        if (flag) {
            printf("Epoch %zu/%zu\tCost: %f\n", e+1, epochs, cost / (xs.cols));
            flag = false;
        }

        // if ((e+1) % _nl_n_epochs == 0) {
        //     cost = 0.f;
        //     for (size_t i = 0; i < xs.cols; i+=batch_size) {
        //         for (size_t b = 0; b < batch_size; ++b) {
        //             nl_mat_get_col(y, ys, i);
        //             nl_model_feed_forward(model, zsa, asa);
        //             cost += nl_mat_cost(losses, asa[model.count-1], y, model.ct);
        //         }
        //     }
        //     printf("Epoch %zu/%zu\tCost: %f\n", e+1, epochs, cost / (xs.cols));
        // }
    }

    arena_destroy(arena);
}

void nl_model_feed_forward(NeuralNet model, Mat *zsa, Mat *asa)
{
    for (size_t i = 1; i < model.count; ++i) {
        nl_mat_dot(zsa[i-1], model.ws[i-1], asa[i-1]);
        nl_mat_add(zsa[i-1], zsa[i-1], model.bs[i-1]);
        nl_mat_act(asa[i], zsa[i-1], model.acts[i-1]);
    }
}

// http://neuralnetworksanddeeplearning.com/chap2.html
void nl_model_backprop(NeuralNet model, Mat ys, Mat *zsa, Mat *asa, Mat *delta_ws, Mat *delta_bs)
{
    Arena arena = arena_new(nl_arena_capacity);

    size_t l = model.count - 1;

    // Only delta not using arena to alloc memory
    Mat delta = nl_mat_alloc(model.layers[l], 1);

    Mat sp;
    Mat asT;
    Mat delta_dot_asT;
    Mat wsT;
    Mat wsT_dot_delta;

    // Calculate delta
    sp = nl_mat_alloc_with_arena(&arena, zsa[l-1].rows, zsa[l-1].cols);
    nl_mat_act_prime(sp, zsa[l-1], model.acts[l-1]);
    nl_mat_cost_prime(delta, asa[l], ys, model.ct);
    nl_mat_mult(delta, delta, sp);

    // Update weights of the output layer
    // transpose of as
    asT = nl_mat_alloc_with_arena(&arena, asa[l-1].cols, asa[l-1].rows);
    // (delta) dot (as.transpose)
    delta_dot_asT = nl_mat_alloc_with_arena(&arena, delta.rows, asT.cols);

    nl_mat_transpose(asT, asa[l-1]);
    nl_mat_dot(delta_dot_asT, delta, asT);
    nl_mat_copy(delta_ws[l-1], delta_dot_asT);

    nl_mat_copy(delta_bs[l-1], delta);

    // Hidden layers
    for (size_t h = l-1; h > 0; --h) {
        arena_reset(&arena);

        sp = nl_mat_alloc_with_arena(&arena, zsa[h-1].rows, zsa[h-1].cols);
        nl_mat_act_prime(sp, zsa[h-1], model.acts[h-1]);

        wsT = nl_mat_alloc_with_arena(&arena, model.ws[(h+1)-1].cols, model.ws[(h+1)-1].rows);
        nl_mat_transpose(wsT, model.ws[(h+1)-1]); // transpose ws[l+1]
        wsT_dot_delta = nl_mat_alloc_with_arena(&arena, wsT.rows, delta.cols);
        nl_mat_dot(wsT_dot_delta, wsT, delta);

        // Update delta (only delta not using arena to alloc memory)
        nl_mat_free(delta);
        delta = nl_mat_alloc(sp.rows, sp.cols);
        nl_mat_mult(delta, wsT_dot_delta, sp);

        // Update weights
        asT = nl_mat_alloc_with_arena(&arena, asa[h-1].cols, asa[h-1].rows); // transpose of as
        nl_mat_transpose(asT, asa[h-1]);

        delta_dot_asT = nl_mat_alloc_with_arena(&arena, delta.rows, asT.cols); // (delta) dot (as.transpose)
        nl_mat_dot(delta_dot_asT, delta, asT);
        nl_mat_copy(delta_ws[h-1], delta_dot_asT);

        nl_mat_copy(delta_bs[h-1], delta);
    }

    nl_mat_free(delta);
    arena_destroy(arena);
}

void nl_model_predict(NeuralNet model, Mat ins, Mat outs)
{
    Arena arena = arena_new(nl_arena_capacity);
    // Alloc memoy for zs, array of column vectors
    Mat *zsa = arena_alloc(&arena, sizeof(Mat) * (model.count - 1));
    for (size_t i = 1; i < model.count; ++i) {
        zsa[i-1] = nl_mat_alloc_with_arena(&arena, model.layers[i], 1);
        nl_mat_zero(zsa[i-1]);
    }

    // Alloc memory for activations, array of column vectors
    Mat *asa = arena_alloc(&arena, sizeof(Mat) * model.count);
    for (size_t i = 0; i < model.count; ++i) {
        asa[i] = nl_mat_alloc_with_arena(&arena, model.layers[i], 1);
    }
    nl_mat_copy(asa[0], ins);

    nl_model_feed_forward(model, zsa, asa);

    // Assign predict result to outs
    nl_mat_copy(outs, asa[model.count-1]);

    // Free memory
    arena_destroy(arena);
}

float nl_model_accuracy_score(Mat y_true, Mat y_predict)
{
    // TODO: Do softmax?
    size_t correct_count = 0;
    size_t predictions = y_predict.cols;

    // For all the predictions
    float y_true_val, y_predict_val;
    size_t y_true_idx, y_predict_idx;
    for (size_t pred = 0; pred < predictions; ++pred) {
        // Find max value in y_true, which is the expected value
        y_true_val = -1.f;
        y_true_idx = 0;
        for (size_t i = 0; i < y_true.rows; ++i) {
            if (NL_MAT_AT(y_true, i, pred) > y_true_val) {
                y_true_val = NL_MAT_AT(y_true, i, pred);
                y_true_idx = i;
            }
        }
        
        // Find max value in y_predict, which is the predicted value
        y_predict_val= -1.f;
        y_predict_idx = 0;
        for (size_t i = 0; i < y_predict.rows; ++i) {
            if (NL_MAT_AT(y_predict, i, pred) > y_predict_val) {
                y_predict_val = NL_MAT_AT(y_predict, i, pred);
                y_predict_idx = i;
            }
        }

        if (y_true_idx == y_predict_idx) {
            correct_count += 1;
        }
    }

    printf("Correct count: %zu\n", correct_count);
    return correct_count / predictions;
}

void nl_model_free(NeuralNet model)
{
    for (size_t i = 1; i < model.count; ++i) {
        nl_mat_free(model.ws[i-1]);
        nl_mat_free(model.bs[i-1]);
    }
    NL_FREE(model.layers);
    NL_FREE(model.acts);
    NL_FREE(model.ws);
    NL_FREE(model.bs);
}

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
    NL_FSCANF(fptr, "%s", buf);
    NL_FSCANF(fptr, "%zu", &(model->count));
    // printf("%zu\n", model->count);

    // Alloc memory
    model->layers = NL_MALLOC(sizeof(size_t) * model->count);
    model->acts = NL_MALLOC(sizeof(size_t) * (model->count - 1));
    model->ws = NL_MALLOC(sizeof(Mat) * (model->count - 1));
    model->bs = NL_MALLOC(sizeof(Mat) * (model->count - 1));

    // Layers
    NL_FSCANF(fptr, "%s", buf);
    for (size_t i = 0; i < model->count; ++i) {
        NL_FSCANF(fptr, "%zu ", &(model->layers[i]));
        printf("%zu ", model->layers[i]);
    }
    printf("\n");

    // Activations
    NL_FSCANF(fptr, "%s", buf);
    for (size_t i = 0; i < model->count - 1; ++i) {
        NL_FSCANF(fptr, "%u ", &(model->acts[i]));
        printf("%u ", model->acts[i]);
    }
    printf("\n");

    // Cost func
    NL_FSCANF(fptr, "%s", buf);
    NL_FSCANF(fptr, "%u", &(model->ct));
    printf("%s\n", buf);
    printf("%u\n", model->ct);
    printf("cost ok\n");

    // Weights
    NL_FSCANF(fptr, "%s", buf);
    for (size_t i = 0; i < model->count - 1; ++i) {
        NL_FSCANF(fptr, "%s", buf);
        model->ws[i] = nl_mat_alloc(model->layers[i + 1], model->layers[i]);
        for (size_t r = 0; r < model->ws[i].rows; ++r) {
            for (size_t c = 0; c < model->ws[i].cols; ++c) {
                NL_FSCANF(fptr, "%f ", &NL_MAT_AT(model->ws[i], r, c));
            }
        }
        NL_FSCANF(fptr, "%s", buf);
    }
    printf("weight ok\n");

    // Biases
    NL_FSCANF(fptr, "%s", buf);
    for (size_t i = 0; i < model->count - 1; ++i) {
        NL_FSCANF(fptr, "%s", buf);
        model->bs[i] = nl_mat_alloc(model->layers[i + 1], 1);
        for (size_t r = 0; r < model->bs[i].rows; ++r) {
            for (size_t c = 0; c < model->bs[i].cols; ++c) {
                NL_FSCANF(fptr, "%f ", &NL_MAT_AT(model->bs[i], r, c));
            }
        }
        NL_FSCANF(fptr, "%s", buf);
    }
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
    NL_FSCANF(fptr, "%s", buf);
    printf("Loading: %s  ", buf);
    NL_FSCANF(fptr, "%zu", &(model->count));
    printf("Loaded\n");

    // Alloc memory
    // model->layers = NL_MALLOC(sizeof(size_t) * model->count);
    model->layers = arena_alloc(arena, sizeof(size_t) * model->count);
    // model->acts = NL_MALLOC(sizeof(size_t) * (model->count - 1));
    model->acts = arena_alloc(arena, sizeof(size_t) * (model->count - 1));
    // model->ws = NL_MALLOC(sizeof(Mat) * (model->count - 1));
    model->ws = arena_alloc(arena, sizeof(Mat) * (model->count - 1));
    model->bs = arena_alloc(arena, sizeof(Mat) * (model->count - 1));

    // Layers
    NL_FSCANF(fptr, "%s", buf);
    printf("Loading: %s  ", buf);
    for (size_t i = 0; i < model->count; ++i) {
        NL_FSCANF(fptr, "%zu ", &(model->layers[i]));
        printf("%zu ", model->layers[i]);
    }
    printf("\n");
    printf("Loaded\n");

    // Activations
    NL_FSCANF(fptr, "%s", buf);
    printf("Loading: %s\n", buf);
    for (size_t i = 0; i < model->count - 1; ++i) {
        NL_FSCANF(fptr, "%u ", &(model->acts[i]));
        printf("%u ", model->acts[i]);
    }
    printf("\n");
    printf("Loaded\n");

    // Cost func
    NL_FSCANF(fptr, "%s", buf);
    NL_FSCANF(fptr, "%u", &(model->ct));
    printf("Loading: %s  ", buf);
    printf("%u\n", model->ct);
    printf("Loaded\n");

    // Weights
    NL_FSCANF(fptr, "%s", buf);
    printf("Loading: %s\n", buf);
    for (size_t i = 0; i < model->count - 1; ++i) {
        NL_FSCANF(fptr, "%s", buf);
        model->ws[i] = nl_mat_alloc_with_arena(arena, model->layers[i + 1], model->layers[i]);
        for (size_t r = 0; r < model->ws[i].rows; ++r) {
            for (size_t c = 0; c < model->ws[i].cols; ++c) {
                NL_FSCANF(fptr, "%f ", &NL_MAT_AT(model->ws[i], r, c));
            }
        }
        NL_FSCANF(fptr, "%s", buf);
    }
    printf("Loaded\n");

    // Biases
    NL_FSCANF(fptr, "%s", buf);
    printf("Loading: %s\n", buf);
    for (size_t i = 0; i < model->count - 1; ++i) {
        NL_FSCANF(fptr, "%s", buf);
        model->bs[i] = nl_mat_alloc_with_arena(arena, model->layers[i + 1], 1);
        for (size_t r = 0; r < model->bs[i].rows; ++r) {
            for (size_t c = 0; c < model->bs[i].cols; ++c) {
                NL_FSCANF(fptr, "%f ", &NL_MAT_AT(model->bs[i], r, c));
            }
        }
        NL_FSCANF(fptr, "%s", buf);
    }
    printf("Loaded\n");

    fclose(fptr);
}

#endif // NERUALIB_IMPLEMENTATION
