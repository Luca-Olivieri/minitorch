#ifndef TENSORS_H
#define TENSORS_H

#include <memory>
#include <vector>
#include <iostream>

#include "tensor_nodes.h"

class Tensor {
public:
    std::shared_ptr<TensorNode> m_node;

    friend class TensorNode;

    Tensor(
        std::vector<size_t> shape,
        float value = 0.0f
    );
    
    Tensor(
        std::shared_ptr<TensorNode> node
    );

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    
    float& operator[](const std::vector<size_t>& md_index);

    float& item();

    bool is_contiguous();

    Tensor grad() const;

    const std::vector<size_t>& shape();

    void backward(
        bool create_graph = false
    );

    void accumulate_grad(
        const Tensor& gradient,
        bool create_graph = false
    );

    std::map<TensorNode*, int> compute_in_degree();

    void topological_backprop(
        std::map<TensorNode*, int>& in_degree,
        bool create_graph
    );
    
    void fill_inplace(
        float value
    );

    Tensor fill(
        float value
    );

    void linspace_inplace(
        float start,
        float end
    );

    Tensor linspace(
        float start,
        float end
    );

    static Tensor linspace(
        std::vector<size_t> shape,
        float start,
        float end
    );

    void reset_grad();
    
    void zero_grad();

    template <auto Op, typename BW_OP, typename... Tensors>
    Tensor apply_op_ag(
        const Tensors&... others
    ) {
        TensorStorage out_storage = Op(m_node->m_storage, others.m_node->m_storage...);
        std::shared_ptr<TensorNode> out = std::make_shared<TensorNode>(std::move(out_storage));
        out->m_bw_op = std::make_unique<BW_OP>(
            m_node,       // First operand
            others...  // others are Tensors
        );

        return Tensor(out);
    }

    Tensor operator+(
        const Tensor& other
    );
    
    void operator+=(
        const Tensor& other
    );

    Tensor operator-();
    
    Tensor operator-(
        const Tensor& other
    );
    
    Tensor operator*(
        const Tensor& other
    );
    
    Tensor operator/(
        const Tensor& other
    );
    
    Tensor pow(
        const Tensor& other
    );
    
    Tensor log();
    
    Tensor sum(
        const size_t dim
    );
    
    Tensor unsqueeze(
        const size_t dim
    ) const;
    
    Tensor squeeze(
        const size_t dim
    ) const;
    
    Tensor repeat(
        const size_t dim,
        const size_t times
    ) const;

    Tensor matmul(
        const Tensor& other
    );
private:
};

#endif
