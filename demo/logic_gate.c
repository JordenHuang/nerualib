#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define TRAIN_SIZE (sizeof(train)/sizeof(train[0]))

// // AND gate
// float train[][3] = {
//     { 0, 0, 0 },
//     { 0, 1, 0 },
//     { 1, 0, 0 },
//     { 1, 1, 1 },
// };

// // OR gate
// float train[][3] = {
//     { 0, 0, 0 },
//     { 0, 1, 1 },
//     { 1, 0, 1 },
//     { 1, 1, 1 },
// };

// NAND gate
float train[][3] = {
    { 0, 0, 1 },
    { 0, 1, 1 },
    { 1, 0, 1 },
    { 1, 1, 0 },
};

// // NOR gate
// float train[][3] = {
//     { 0, 0, 1 },
//     { 0, 1, 0 },
//     { 1, 0, 0 },
//     { 1, 1, 0 },
// };


float rand_float()
{
    return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

float activation_func_derivative(float a)
{
    return a * (1.f - a);
}

float loss_func(float a, float expected)
{
    float d = a - expected;
    return d * d;
}

// float cost_func(float w1, float w2)
// {
//     float total_loss = 0.f;
//     for (size_t i = 0; i < TRAIN_SIZE; ++i) {
//         float z = w1 * train[i][0] + w2 * train[i][1];
//         float a = sigmoidf(z);
//         float loss = loss_func(a, train[i][2]);
//         total_loss += loss;
//     }
//     return total_loss / TRAIN_SIZE;
// }


/**
 * Use batch gradient descent
*/
int main(void)
{
    srand(time(NULL));
    float w1 = rand_float(), w2 = rand_float();
    float b = rand_float();
    float lr = 1e-1;

    for (size_t r = 0; r < 100 * 1000; ++r) {
        float total_cost = 0.f;
        float grad_w1_total = 0.f;
        float grad_w2_total = 0.f;
        float grad_b_total = 0.f;
        for (size_t i = 0; i < TRAIN_SIZE; ++i) {
            float x1 = train[i][0];
            float x2 = train[i][1];
            float expected = train[i][2];
            float z = w1 * x1 + w2 * x2 + b;
            float a = sigmoidf(z);
            float loss = loss_func(a, expected);
            total_cost += loss;
            // d(loss)/d(w1) = d(loss)/d(a) * d(a)/d(z) * d(z)/d(w1)
            float grad_w1 = 2*(a - expected) * activation_func_derivative(a) * x1;
            float grad_w2 = 2*(a - expected) * activation_func_derivative(a) * x2;
            float grad_b = 2*(a - expected) * activation_func_derivative(a) * 1;
            grad_w1_total += grad_w1;
            grad_w2_total += grad_w2;
            grad_b_total += grad_b;
        }
        total_cost /= TRAIN_SIZE;
        grad_w1_total /= TRAIN_SIZE;
        grad_w2_total /= TRAIN_SIZE;
        grad_b_total /= TRAIN_SIZE;

        w1 -= lr * grad_w1_total;
        w2 -= lr * grad_w2_total;
        b -= lr * grad_b_total;

        if (r % 10000 == 0) {
            printf("Epoch: %zu, w1: %f, w2: %f, b: %f, Cost: %f\n", r, w1, w2, b, total_cost);
        }
    }

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            printf("%zu & %zu = %f\n", i, j, sigmoidf(w1*i + w2*j + b));
        }
    }

    return 0;
}
