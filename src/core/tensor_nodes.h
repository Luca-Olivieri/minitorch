#ifndef TENSOR_NODES_H
#define TENSOR_NODES_H

#include <map>

#include "tensor_storages.h"
// #include "tensors.h"

class Tensor;

class BackwardOp;

class TensorNode : public std::enable_shared_from_this<TensorNode> {
public:
    TensorStorage m_storage;
    
    std::unique_ptr<BackwardOp> m_bw_op { nullptr };
    std::shared_ptr<Tensor> m_grad { nullptr };

    TensorNode(
        std::vector<size_t> shape,
        float value = 0.0f
    );

    TensorNode(
        TensorStorage storage
    );

    ~TensorNode();

    friend std::ostream& operator<<(std::ostream& os, const TensorNode& tensor);
    
    float& operator[](const std::vector<size_t>& md_index);
};

#endif
