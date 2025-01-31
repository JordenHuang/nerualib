CC=gcc
CFLAGS=-Wall -Wextra
LIBS=-lm


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

.PHONY: cleann
clean:
	rm *.out
