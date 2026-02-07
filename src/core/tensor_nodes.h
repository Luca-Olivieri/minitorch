#ifndef TENSOR_NODES_H
#define TENSOR_NODES_H

#include "tensor_storages.h"

class BackwardOp;

class TensorNode : public std::enable_shared_from_this<TensorNode> {
public:
    TensorStorage m_value;
    
    std::shared_ptr<BackwardOp> m_bw_op { nullptr };
    std::shared_ptr<TensorNode> m_grad { nullptr };
    bool m_requires_grad { true };

    TensorNode(
        std::vector<size_t> shape
    );

    friend std::ostream& operator<<(std::ostream& os, const TensorNode& tensor);
    
    float& operator[](const std::vector<size_t>& md_index);

    float& item();
    
    void fill(
        float value
    );

    void linspace(
        float start,
        float end
    );

    bool is_contiguous();
    
    template <auto Op, typename BW_OP, typename... Tensors>
    std::shared_ptr<TensorNode> apply_op_ag(
        Tensors&... others
    ) {
        std::shared_ptr<TensorNode> out {std::make_shared<TensorNode>(m_value.m_shape)};
        out->m_value = Op(m_value, others.m_value...);
        out->m_bw_op = std::make_shared<BW_OP>(
            shared_from_this(),       // First operand
            others.shared_from_this()...  // Expand references to pointers
        );

        return out;
    }
    
    std::shared_ptr<TensorNode> operator+(
        TensorNode& other
    );
    
    void operator+=(
        TensorNode& other
    );
    
    std::shared_ptr<TensorNode> operator-();
    
    std::shared_ptr<TensorNode> operator*(
        TensorNode& other
    );
    
    std::shared_ptr<TensorNode> pow(
        TensorNode& other
    );

    void reset_grad();
    
    void zero_grad();

    void backward();

    void backprop();
};

#endif
