# Log Book

1. I started implementing the backpropagation graph logic. Leaf nodes and operations are extensions of a Node class and implement logic to backpropagate starting from the topmost node.
2. I recognized that I do know how to feed this logic to a tensors. Hence, I started implementing tensors.
3. I realised I do not have a testing framework. I understood that Catch2 and Google Test are the standard way to go, but I do not want to integrate external libraries for something so simple, so I will make my own testing functions.
