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
            const std::vector<size_t> shape,
            const float value = 0.0f,
            const bool requires_grad = true
    );
    
    Tensor(
            const std::shared_ptr<TensorNode> node
    );

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    
    float& operator[](
            const std::vector<size_t>& md_index
    ) const;

    float& item() const;

    bool is_contiguous() const;

    Tensor grad() const;

    const std::vector<size_t>& shape() const;
    
    size_t numel() const;

    void backward(
            const bool retain_graph = false
    );

    void accumulate_grad(
            const Tensor& gradient
    );

    std::map<TensorNode*, int> compute_in_degree() const;

    void topological_backprop(
            std::map<TensorNode*, int>& in_degree,
            const bool retain_graph
    ) const;
    
    void fill_inplace(
            const float value
    );

    Tensor fill(
            const float value
    ) const;

    void linspace_inplace(
            const float start,
            const float end
    );

    Tensor linspace(
            const float start,
            const float end
    ) const;

    static Tensor linspace(
            const std::vector<size_t>& shape,
            const float start,
            const float end,
            const bool requires_grad = true
    );

    void reset_grad();
    
    void zero_grad();

    void detach_inplace();

    static bool compute_requires_grad_from_operands(
        const std::vector<Tensor>& others
    );

    template <auto Op, typename GradFn_T, typename... Tensors>
    static Tensor apply_op_ag(
            const Tensors&... operands
    ) {
        TensorStorage out_storage = Op(operands.m_node->m_storage...);
        std::shared_ptr<TensorNode> out = std::make_shared<TensorNode>(
            std::move(out_storage),
            compute_requires_grad_from_operands({operands...})
        );
        if constexpr (!std::is_same_v<GradFn_T, void>) {
            if (out->m_requires_grad) {
                out->m_grad_fn = std::make_unique<GradFn_T>(operands...);
            }
        }
        return Tensor(out);
    }

    Tensor operator+(
            const Tensor& other
    ) const;
    
    void operator+=(
            const Tensor& other
    );
    
    void operator-=(
            const Tensor& other
    );

    Tensor operator-() const;
    
    Tensor operator-(
            const Tensor& other
    ) const;
    
    Tensor operator*(
            const Tensor& other
    ) const;
    
    Tensor operator*(
            float scalar
    ) const;
    
    Tensor operator/(
            const Tensor& other
    ) const;
    
    Tensor pow(
            const Tensor& other
    ) const;
    
    Tensor log() const;

    static Tensor maximum(
            const Tensor& a,
            const Tensor& b
    );

    Tensor operator>(
            const Tensor& other
    ) const;
    
    Tensor operator>=(
            const Tensor& other
    ) const;
    
    Tensor operator<=(
            const Tensor& other
    ) const;
    
    Tensor sum(
            const size_t dim
    ) const;
    
    Tensor mean(
            const size_t dim
    ) const;
    
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
    
    Tensor expand(
            const size_t dim,
            const size_t times
    ) const;

    Tensor one_hot(
        size_t num_classes
    ) const;
    
    Tensor clone() const;

    static Tensor matmul(
            const Tensor& a,
            const Tensor& b
    );
private:
};

namespace mt {
    Tensor stack(
        const std::vector<Tensor>& tensors
    );
}

#endif
