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

#define NL_ARRAY_LEN(arr) (sizeof(arr) / sizeof(arr[0]))
#define NL_MAT_INDEX(columns, r, c) ((r)*(columns) + (c))
#define NL_MAT_AT_INDEX(m, idx) m.items[idx]
#define NL_MAT_AT(m, r, c) m.items[NL_MAT_INDEX(m.cols, r, c)]


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

void nl_rand_init(size_t use_seed, size_t seed);
float nl_rand_float(void);
Mat nl_mat_alloc(size_t row, size_t col);
Mat nl_mat_alloc_with_arena(Arena *arena, size_t row, size_t col);
void nl_mat_zero(Mat m);
void nl_mat_one(Mat m);
void nl_mat_rand(Mat m);
void _nl_mat_print(Mat m, const char *name, size_t padding);
#define nl_mat_print(m) _nl_mat_print(m, #m, 0);
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
void nl_model_train(NeuralNet model, Mat xs, Mat ys, float lr);
void nl_model_feed_forward(NeuralNet model, Mat *zsa, Mat *asa);
void nl_model_backprop(NeuralNet model, Mat ys, Mat *zsa, Mat *asa, float lr);
void nl_model_predict(NeuralNet model, Mat ins, Mat outs);
void nl_model_free(NeuralNet model);

#endif // _NERUALIB_H_

#ifdef NERUALIB_IMPLEMENTATION

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

Arena arena_new(size_t bytes)
{
    Arena a;
    a.capacity = bytes;
    a.size = 0;
    a.begin = malloc(bytes);
    return a;
}

void arena_destroy(Arena a)
{
    free(a.begin);
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


void nl_rand_init(size_t use_seed, size_t seed)
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
           NL_MAT_AT(m, i, j) = nl_rand_float();
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

void nl_model_train(NeuralNet model, Mat xs, Mat ys, float lr)
{
    Arena arena = arena_new(4 * 1024 * 1024);

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
    nl_mat_copy(asa[0], xs);

    // Forward pass
    nl_model_feed_forward(model, zsa, asa);

    Mat losses = nl_mat_alloc_with_arena(&arena, asa[model.count-1].rows, asa[model.count-1].cols);
    float cost = nl_mat_cost(losses, asa[model.count-1], ys, MSE);
    int print_threshold = 1;
    int ran_num = rand() % 100000;
    if (ran_num < print_threshold) {
        printf("  Cost: %f\n", cost);
    }
    // nl_mat_print(losses);
    // nl_mat_free(losses);
    // printf("\n");

    // Backward pass (backpropagaton)
    nl_model_backprop(model, ys, zsa, asa, lr);

    // Free memory
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
void nl_model_backprop(NeuralNet model, Mat ys, Mat *zsa, Mat *asa, float lr)
{
    Arena arena = arena_new(4 * 1024 * 1024);

    size_t l = model.count - 1;

    // Only delta not using arena to alloc memory
    Mat delta = nl_mat_alloc(model.layers[l], 1);

    Mat sp;
    Mat asT;
    Mat delta_dot_asT;
    Mat delta_mult_lr;
    Mat wsT;
    Mat wsT_dot_delta;

    // Calculate delta
    sp = nl_mat_alloc_with_arena(&arena, zsa[l-1].rows, zsa[l-1].cols);
    nl_mat_act_prime(sp, zsa[l-1], model.acts[l-1]);
    nl_mat_cost_prime(delta, asa[l], ys, model.ct);
    nl_mat_mult(delta, delta, sp);
    // nl_mat_free(sp);

    // Update weights of the output layer
    // transpose of as
    asT = nl_mat_alloc_with_arena(&arena, asa[l-1].cols, asa[l-1].rows);
    // (delta) dot (as.transpose)
    delta_dot_asT = nl_mat_alloc_with_arena(&arena, delta.rows, asT.cols);

    nl_mat_transpose(asT, asa[l-1]);
    nl_mat_dot(delta_dot_asT, delta, asT);
    nl_mat_mult_c(delta_dot_asT, delta_dot_asT, -lr);
    nl_mat_add(model.ws[l-1], model.ws[l-1], delta_dot_asT);
    /*
    Mat *temps = arena_alloc(&arena, sizeof(Mat) * 3);

    temps[0] = nl_mat_alloc(asa[l-1].cols, asa[l-1].rows);
    temps[1] = nl_mat_alloc(delta.rows, temps[0].cols);
    nl_mat_transpose(temps[0], asa[l-1]);
    nl_mat_dot(temps[1], delta, temps[0]);
    nl_mat_mult_c(temps[1], temps[1], -lr);
    nl_mat_add(model.ws[l-1], model.ws[l-1], temps[1]);
    */

    // Update bias of the output layer
    delta_mult_lr = nl_mat_alloc_with_arena(&arena, delta.rows, delta.cols);
    nl_mat_mult_c(delta_mult_lr, delta, -lr);
    nl_mat_add(model.bs[l-1], model.bs[l-1], delta_mult_lr);
    /*
    nl_mat_free(temps[0]);
    temps[0] = nl_mat_alloc(delta.rows, delta.cols);
    nl_mat_mult_c(temps[0], delta, -lr);
    nl_mat_add(model.bs[l-1], model.bs[l-1], temps[0]);
    */

    arena_reset(&arena);

    // Hidden layers
    for (size_t h = l-1; h > 0; --h) {
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

        delta_mult_lr = delta_dot_asT;
        nl_mat_mult_c(delta_mult_lr, delta_dot_asT, -lr);
        nl_mat_add(model.ws[(h)-1], model.ws[(h)-1], delta_mult_lr);

        // Update bias
        delta_mult_lr = nl_mat_alloc_with_arena(&arena, delta.rows, delta.cols);
        nl_mat_mult_c(delta_mult_lr, delta, -lr);
        nl_mat_add(model.bs[(h)-1], model.bs[(h)-1], delta_mult_lr);


        arena_reset(&arena);
        /*
        Mat sp = nl_mat_alloc(zsa[h-1].rows, zsa[h-1].cols);
        nl_mat_act_prime(sp, zsa[h-1], act);
        temps[2] = nl_mat_alloc(model.ws[(h+1)-1].cols, model.ws[(h+1)-1].rows);
        nl_mat_transpose(temps[2], model.ws[(h+1)-1]); // transpose ws[l+1]
        temps[3] = nl_mat_alloc(temps[2].rows, delta.cols);
        nl_mat_dot(temps[3], temps[2], delta);
        nl_mat_free(delta);
        delta = nl_mat_alloc(sp.rows, sp.cols);
        nl_mat_mult(delta, temps[3], sp);
        nl_mat_free(sp);

        // Update weights
        temps[4] = nl_mat_alloc(asa[h-1].cols, asa[h-1].rows); // transpose of as
        temps[5] = nl_mat_alloc(delta.rows, temps[4].cols);    // (delta) dot (as.transpose)
        nl_mat_transpose(temps[4], asa[h-1]);
        nl_mat_dot(temps[5], delta, temps[4]);
        nl_mat_mult_c(temps[5], temps[5], -lr);
        nl_mat_add(model.ws[(h)-1], model.ws[(h)-1], temps[5]);

        // Update bias
        nl_mat_free(temps[4]);
        temps[4] = nl_mat_alloc(delta.rows, delta.cols);
        nl_mat_mult_c(temps[4], delta, -lr);
        nl_mat_add(model.bs[(h)-1], model.bs[(h)-1], temps[4]);

        for (size_t j = 2; j <= 5; ++j) {
            nl_mat_free(temps[j]);
        }
        */
    }

    /*
    nl_mat_free(temps[0]);
    nl_mat_free(temps[1]);
    */
    arena_destroy(arena);
}

void nl_model_predict(NeuralNet model, Mat ins, Mat outs)
{
    // Alloc memoy for zs, array of column vectors
    Mat *zsa = NL_MALLOC(sizeof(Mat) * (model.count - 1));
    for (size_t i = 1; i < model.count; ++i) {
        zsa[i-1] = nl_mat_alloc(model.layers[i], 1);
        nl_mat_zero(zsa[i-1]);
    }
    // Alloc memory for activations, array of column vectors
    Mat *asa = NL_MALLOC(sizeof(Mat) * model.count);
    for (size_t i = 0; i < model.count; ++i) {
        asa[i] = nl_mat_alloc(model.layers[i], 1);
    }
    nl_mat_copy(asa[0], ins);

    for (size_t i = 1; i < model.count; ++i) {
        nl_mat_dot(zsa[i-1], model.ws[i-1], asa[i-1]);
        nl_mat_add(zsa[i-1], zsa[i-1], model.bs[i-1]);
        nl_mat_act(asa[i], zsa[i-1], model.acts[i-1]);
    }

    // Assign predict result to outs
    nl_mat_copy(outs, asa[model.count-1]);

    // Free memory
    for (size_t i = 0; i < model.count - 1; ++i) {
        nl_mat_free(zsa[i]);
    }
    NL_FREE(zsa);

    for (size_t i = 0; i < model.count; ++i) {
        nl_mat_free(asa[i]);
    }
    NL_FREE(asa);
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

#endif // NERUALIB_IMPLEMENTATION
