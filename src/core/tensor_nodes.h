#ifndef TENSOR_NODES_H
#define TENSOR_NODES_H

#include <map>

#include "tensor_storages.h"

class Tensor;
class GradFn;

class TensorNode : public std::enable_shared_from_this<TensorNode> {
public:
    TensorStorage m_storage;
    
    std::unique_ptr<GradFn> m_grad_fn { nullptr };
    std::shared_ptr<Tensor> m_grad { nullptr };

    TensorNode(
        const std::vector<size_t> shape,
        const float value = 0.0f
    );

    TensorNode(
        TensorStorage&& storage
    );
};

#endif
