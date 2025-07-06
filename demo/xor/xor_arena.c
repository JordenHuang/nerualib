#include <stdio.h>

#define NERUALIB_IMPLEMENTATION
#include "neuralib.h"

int main(void) {
    printf("[INFO] Set some parameters\n");
    // Set seed, to reproduce the same result
    nl_rand_init(true, 20250706);
    NL_PRINT_COST_EVERY_N_EPOCHS(1000);

    printf("[INFO] Allocate space\n");
    Arena arena_train = arena_new(1024);

    printf("[INFO] Prepare training data\n");
    Mat train_x = (Mat){
        .rows = 2,
        .cols = 4,
        .items = (float[]){
            0, 0, 1, 1,
            0, 1, 0, 1
        }
    };

    Mat train_y = (Mat){
        .rows = 1,
        .cols = 4,
        .items = (float[]){
            0, 1, 1, 0
        }
    };

    printf("[INFO] Define model layer components\n");
    NeuralNet model;
    size_t layers[] = {2, 2, 1};
    Activation_type acts[] = {SIGMOID, SIGMOID};
    nl_define_layers_with_arena(&arena_train, &model, NL_ARRAY_LEN(layers), layers, acts, MSE);

    printf("[INFO] View the model structure\n");
    nl_model_summary(model, stdout);

    printf("[INFO] Set hyperparameters\n");
    float lr = 2e-1;
    size_t epochs = 5000;
    size_t batch_size = 1;

    printf("[INFO] Train\n");
    nl_model_train(model, train_x, train_y, lr, epochs, batch_size, true);

    printf("[INFO] Validation\n");
    Mat vx = nl_mat_alloc_with_arena(&arena_train, 2, 1);
    Mat vy = nl_mat_alloc_with_arena(&arena_train, 1, 1);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            NL_MAT_AT(vx, 0, 0) = i;
            NL_MAT_AT(vx, 1, 0) = j;
            nl_model_predict(model, vx, vy);
            printf("%zu ^ %zu = %.2f , expect: %zu",
                   i,
                   j,
                   NL_MAT_AT(vy, 0, 0),
                   i ^ j);

            float dist = (float)(i^j) - NL_MAT_AT(vy, 0, 0);
            if (dist < 0.f)
                dist = -dist;
            printf(" [%c]\n", dist < 0.5 ? 'V' : ' ');
        }
    }

    printf("[INFO] Saveing model\n");
    char reply;
    printf("Save model? (y/n) ");
    scanf("%c", &reply);
    if (reply == 'y') {
        const char model_path[] = "myModel.model";
        nl_model_save(model_path, model);
        printf("Model [%s] saved\n", model_path);
    }

    printf("[INFO] Freeing allocated space\n");
    arena_destroy(arena_train);

    return 0;
}
