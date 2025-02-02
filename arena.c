#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

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


int main(void)
{
    Arena a = arena_new(1024);
    arena_info(a);

    printf("----------\n");
    const size_t sz = 10;
    float *arr_f = arena_alloc(&a, sizeof(float) * sz);
    for (size_t i = 0; i < sz; ++i) {
        arr_f[i] = i;
    }
    for (size_t i = 0; i < sz; ++i) {
        printf("arr_f[%zu] = %f\n", i, arr_f[i]);
    }
    printf("----------\n");

    arena_info(a);
    printf("----------\n");

    printf("[INFO] Reset\n");
    arena_reset(&a);
    arena_info(a);
    printf("----------\n");

    int *arr_i = arena_alloc(&a, sizeof(int) * sz/2);
    for (size_t i = 0; i < sz; ++i) {
        arr_i[i] = i;
    }
    for (size_t i = 0; i < sz; ++i) {
        printf("arr_i[%zu] = %d\n", i, arr_i[i]);
    }
    printf("----------\n");

    arena_info(a);
    printf("----------\n");

    arena_destroy(a);
    return 0;
}


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
    }
}

void arena_reset(Arena *a)
{
    a->size = 0;
}
