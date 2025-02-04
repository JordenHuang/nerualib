CC=gcc
CFLAGS=-Wall -Wextra
LIBS=-lm -I../ -I./thirdparty


.PHONY: all clean

main:
	$(CC) $(CFLAGS) -o main.out main.c $(LIBS)

test:
	$(CC) $(CFLAGS) -o test.out test.c $(LIBS)

# From older to newer
mul:
	$(CC) $(CFLAGS) -o mul.out mul.c $(LIBS)

logic_gate:
	$(CC) $(CFLAGS) -o logic_gate.out logic_gate.c $(LIBS)

xor:
	$(CC) $(CFLAGS) -o xor.out xor.c $(LIBS)

one_bit_adder:
	$(CC) $(CFLAGS) -o one_bit_adder.out one_bit_adder.c $(LIBS)

image_upscale:
	$(CC) $(CFLAGS) -o image_upscale.out image_upscale.c $(LIBS)


clean:
	rm -f *.out *.out

