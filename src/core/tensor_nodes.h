#ifndef TENSOR_NODES_H
#define TENSOR_NODES_H

#include "tensor_storages.h"

class BackwardOp;

class TensorNode {
public:
    TensorStorage m_value;
    
    std::shared_ptr<BackwardOp> m_bw_op { nullptr };
    std::shared_ptr<TensorNode> m_grad { nullptr };
    bool m_requires_grad { true };

    TensorNode(
        std::vector<size_t> shape
    );

    // TODO provide a constructor which accepts a TensorStorage

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
    TensorNode apply_op_ag(
        Tensors&... others
    ) const {
        TensorNode out{m_value.m_shape};
        out.m_value = Op(m_value, others.m_value...);
        out.m_bw_op = std::make_shared<BW_OP>(
            const_cast<TensorNode*>(this),       // First operand
            const_cast<TensorNode*>(&others)...  // Expand references to pointers
        );

        return out;
    }
    
    TensorNode operator+(
        const TensorNode& other
    ) const;
    
    void operator+=(
        const TensorNode& other
    );
    
    TensorNode operator-() const;
    
    TensorNode operator*(
        const TensorNode& other
    ) const;
    
    TensorNode pow(
        const TensorNode& other
    ) const;

    void reset_grad();
    
    void zero_grad();

    void backward();

    void backprop();
};

#endif
