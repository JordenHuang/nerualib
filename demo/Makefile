CC         = gcc
CFLAGS     = -Wall -Wextra -O2
LIBS       = -lm -I../
THIRDPARTY = -I./thirdparty

# List of available projects (directories containing source code)
PROJECTS = image_upscaler coordinate_based_network one_bit_adder xor mul logic_gate test

.PHONY: all clean $(PROJECTS)

all:
	@echo "Usage: make <project_name>"
	@echo "Available projects: $(PROJECTS)"

test:
	$(CC) $(CFLAGS) -o $@.out $@.c $(LIBS)

logic_gate mul:
	$(CC) $(CFLAGS) -o old/$@.out old/$@.c $(LIBS)

# Porject that needs only the neuralib.h
xor one_bit_adder:
	$(CC) $(CFLAGS) -o $@/$@.out $@/$@.c $(LIBS)

# Project that needs thirdparty dependencies
coordinate_based_network:
	$(CC) $(CFLAGS) -o $@/$@.out $@/$@.c $(LIBS) $(THIRDPARTY)

image_upscaler:
	$(CC) $(CFLAGS) -o $@/$@.out $@/$@.c $(LIBS) $(THIRDPARTY)
	$(CC) $(CFLAGS) -o $@/model_load_and_score.out $@/model_load_and_score.c $(LIBS) $(THIRDPARTY)

clean:
	rm -f **/*.out

