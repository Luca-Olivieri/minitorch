#ifndef TENSORS_H
#define TENSORS_H

#include "tensor_nodes.h"

class Tensor {
    std::shared_ptr<TensorNode> node;
    
    Tensor(
        std::vector<size_t> shape
    );
};


#endif
