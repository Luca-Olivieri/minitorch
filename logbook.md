# Log Book

- 2026-01-15: 
    I started implementing the backpropagation graph logic. Leaf nodes and operations are extensions of a Node class and implement logic to backpropagate starting from the topmost node.
- 2026-01-15:
    I recognized that I do know how to feed this logic to a tensors. Hence, I started implementing tensors.
- 2026-01-15:
    I realised I do not have a testing framework. I understood that Catch2 and Google Test are the standard way to go, but I do not want to integrate external libraries for something so simple, so I will make my own testing functions.
- 2026-01-31: 
    I implemented reshape(), transpose() and slice() tensor methods, which operator solely on the strides without touching the array memory.
    However, to iterate through restrided data in a clean way, I need to use some custom iterators, which have some overhead. To speed this up, ChatGPT told me to create a stride-counter iterator, to collapse contiguous dimensions, and to create logic to make restrided data contiguous (creating new data). I will do this in a second moment, I will keep using this naive but correct implementation and see how to optimize it later on.
- 2026-01-31: implemented the dice() tensor methods, reindexing operators are easier to implement than expected.
