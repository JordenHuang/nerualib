# History

## 2025/02/19
- Add model save and model load in coordinate_based_network.c, but not yet merge into neuralib.h
## 2025/02/16
- Remove image upscaling(/demo/image_upscale.c)
- Re-structure the demo projects

## 2025/02/04
- Try to do image upscaling(/demo/image_upscale.c), but it only take a small image, and no matter what the content is in that image, it will always generate a output image that look like the training image, so I think it's not useful, and I decide to try coordinate based neural network to perform image upscaling.
- Mini-batch gradient descent
- Use arena to manage memories

## 2025/02/02
- Add one_bit_adder, 3 input neurons, 1 hidden layer with 4 neurons, and an output layer with 2 neurons
- Update neuralib.h to use arena to easily manage the memory

## 2025/01/31
- Add neuralib.h, a framework for nerual network
- Add xor.c, 1 input layer, 1 hidden layer with 2 neurons, 1 output layer

## 2025/01/25
- Add main.c (currently logic_gates.c), a single neural with two input and one output. It is basically an AND gate (or OR/NAND/NOR gate)
- Add mul.c, a single neural with one input and one output. It multiplies your input by 3
