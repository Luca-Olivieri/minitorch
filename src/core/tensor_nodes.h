#ifndef TENSOR_NODES_H
#define TENSOR_NODES_H

#include <map>

#include "tensor_storages.h"
#include "tensors.h"

class BackwardOp;

class TensorNode : public std::enable_shared_from_this<TensorNode> {
public:
    TensorStorage m_storage;
    
    std::shared_ptr<BackwardOp> m_bw_op { nullptr };
    std::shared_ptr<TensorNode> m_grad { nullptr };
    bool m_requires_grad { true };

    TensorNode(
        std::vector<size_t> shape
    );

    TensorNode(
        TensorStorage& storage
    );

    // non-copyable object
    TensorNode(const TensorNode&) = delete;
    TensorNode& operator=(const TensorNode&) = delete;
    TensorNode(TensorNode&&) = default;
    TensorNode& operator=(TensorNode&&) = default;

    friend std::ostream& operator<<(std::ostream& os, const TensorNode& tensor);
    
    float& operator[](const std::vector<size_t>& md_index);

    float& item();
    
    void fill_inplace(
        float value
    );

    void linspace_inplace(
        float start,
        float end
    );

    bool is_contiguous();
    
    template <auto Op, typename BW_OP, typename... Tensors>
    Tensor apply_op_ag(
        const Tensors&... others
    ) {
        TensorStorage out_storage = Op(m_storage, others.m_node->m_storage...);
        std::shared_ptr<TensorNode> out {
            std::make_shared<TensorNode>(out_storage)
        };
        out->m_bw_op = std::make_shared<BW_OP>(
            Tensor(shared_from_this()),       // First operand
            others...  // others are Tensors
        );

        return Tensor(out);
    }

    Tensor reshape(
        std::vector<size_t> shape
    );

    Tensor transpose(
        size_t dim_1,
        size_t dim_2
    );

    
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
    
    Tensor pow(
        const Tensor& other
    );
    
    Tensor log();

    void reset_grad();
    
    void zero_grad();

    std::map<TensorNode*, int> compute_in_degree();

    void topological_backprop(
        std::map<TensorNode*, int>& in_degree,
        bool create_graph
    );

    void backward(
        bool create_graph = false
    );
    
    void backprop(
        bool create_graph = false
    );

    void accumulate_grad(
        const Tensor& gradient,
        bool create_graph = false
    );
};

#endif
