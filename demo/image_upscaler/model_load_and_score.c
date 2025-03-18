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
    Arena arena = arena_new(64 * 1024 * 1024);
    NeuralNet model;

    const char model_path[] = "./image_upscaler/upscaler.model";
    nl_model_load_with_arena(&arena, model_path, &model);
    printf("Model loaded\n");

    nl_model_summary(model, stdout);

    // nl_model_accuracy_score();

    arena_destroy(arena);
    return 0;
}
