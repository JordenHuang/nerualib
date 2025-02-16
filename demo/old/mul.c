#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define TRAIN_SIZE (sizeof(train)/sizeof(train[0]))
size_t train[][2] = {
    { 1, 3 },
    { 2, 6 },
    { 3, 9 },
    { 4, 12 },
};

// float activation_func(float x)
// {
//     return 1.f / (1.f + expf(-x));
// }

float loss_func(float a, float expected)
{
    float d = a - expected;
    return d * d;
}

// float activation_func_derivative(float a)
// {
//     return a * (1.f - a);
// }

float rand_float()
{
    return (float)rand() / (float)RAND_MAX;
}

enum GD {
    Stochastic_Gradient_Descent,
    Batch_Gradient_Descent,
    // Mini_Batch_Gradient_Descent,
};

int main(void)
{
    srand(0);
    size_t x1;
    float w1 = rand_float();
    float lr = 1e-2;

    enum GD gradient_descent_method = Stochastic_Gradient_Descent; //Batch_Gradient_Descent;

    if (gradient_descent_method == Stochastic_Gradient_Descent) {
        for (size_t r = 0; r < 100; ++r) {
            float cost = 0;
            for (size_t i = 0; i < TRAIN_SIZE; ++i) {
                x1 = train[i][0];
                float out = x1 * w1;
                // float a = activation_func(out);
                float a = out;
                float loss = loss_func(a, train[i][1]);
                cost += loss;

                // d(loss)/d(w1) = d(loss)/d(a) * d(a)/d(out) * d(out)/d(w1)
                // float grad_w1 = 2 * (a - train[i][1]) * activation_func_derivative(a) * x1;
                float grad_w1 = 2 * (a - train[i][1]) * 1.0f * x1;
                w1 -= lr * grad_w1;

            }

            if (r % 10 == 0) {
                printf("Epoch: %zu, w1: %f, Loss: %f\n", r, w1, cost);
            }
        }
    } else if (gradient_descent_method == Batch_Gradient_Descent) {
        for (size_t r = 0; r < 100; ++r) {
            float cost = 0;
            float grad_w1_total = 0;
            for (size_t i = 0; i < TRAIN_SIZE; ++i) {
                x1 = train[i][0];
                float out = x1 * w1;
                // float a = activation_func(out);
                float a = out;
                float loss = loss_func(a, train[i][1]);
                cost += loss;
                float grad_w1 = 2 * (a - train[i][1]) * 1.0f * x1;
                grad_w1_total += grad_w1;
            }
            cost /= TRAIN_SIZE;
            grad_w1_total /= TRAIN_SIZE;

            w1 -= lr * grad_w1_total;

            if (r % 10 == 0) {
                printf("Epoch: %zu, w1: %f, Loss: %f\n", r, w1, cost);
            }
        }
    }

    printf("----------\n");
    printf("Final w1: %f\n", w1);

    printf("----------\n");
    float num;
    printf("Enter a number: ");
    scanf("%f", &num);
    printf("Result: %f\n", (num * w1));

    return 0;
}
