#ifndef BACKWARD_OPS
#define BACKWARD_OPS

#include <iostream>
#include <memory>
#include <vector>
#include <functional>

#include "tensor_nodes.h"
#include "tensors.h"

class BackwardOp {
public:
    BackwardOp();

    virtual ~BackwardOp() = default;

    virtual std::ostream& print(std::ostream& os) const = 0;
    friend std::ostream& operator<<(std::ostream& os, const BackwardOp& op);

    // virtual void init_operands_grad_if_none() = 0;
    
    virtual void reset_all_grads() = 0;
    
    virtual void compute_operands_grad(const Tensor& out, bool create_graph = false) = 0;

    virtual std::vector<Tensor> get_operands() const = 0;
};

template <size_t N>
class NBackwardOp: public BackwardOp {
public:
    static constexpr size_t s_N = N; // expose N as a static member
    std::array<Tensor, N> m_operands;

        // 1. Variadic Template Constructor
    template <typename... Tensors>
    explicit NBackwardOp(
        Tensors... operands
    ): m_operands{{operands...}} { // Optimization note: Move semantics (Tensors&&... and std::forward) could be used here for efficiency to avoid atomic ref-count increments on shared_ptr
        // 2. Compile-time Arity Check
        static_assert(sizeof...(Tensors) == N, 
            "Error: Number of arguments provided to constructor must match template parameter N.");
            
        // 3. (Optional) Strict Type Checking using C++17 Fold Expressions
        // This ensures every argument is actually a Tensor
        static_assert((std::is_convertible_v<Tensors, Tensor> && ...), 
            "Error: All arguments must be implicitly convertible to Tensor.");
    }

    /*
    void init_operands_grad_if_none() {
        for (size_t i {0}; i<N; i++) {
            Tensor& operand = m_operands[i];

            if (!operand.m_node->m_grad) {
                operand.m_node->m_grad = std::make_shared<TensorNode>(operand.shape());
                operand.m_node->m_grad->m_requires_grad = false;
            }
        }
    }
    */

    void reset_all_grads() override {
        for (size_t i {0}; i<N; i++) {
            Tensor& operand = m_operands[i];
            operand.zero_grad();
        }
    }

    std::vector<Tensor> get_operands() const override {
        std::vector<Tensor> ops;
        ops.reserve(N);
        for(const auto& op : m_operands) {
            ops.push_back(op);
        }
        return ops;
    }
};

class BackwardAdd : public NBackwardOp<2> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardMinus : public NBackwardOp<1> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardSub : public NBackwardOp<2> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardMult : public NBackwardOp<2> {
public:    
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardPow : public NBackwardOp<2> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardDiv : public NBackwardOp<2> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardLog : public NBackwardOp<1> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardReduce : public NBackwardOp<1> {
public:
    using NBackwardOp<s_N>::NBackwardOp;
};

class BackwardSum : public BackwardReduce {
public:
    size_t m_dim;
    size_t m_original_times;
    
    BackwardSum(
        Tensor reduced_tensor,
        const size_t dim,
        const size_t original_times
    );

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardView : public NBackwardOp<1> {
public:
    using NBackwardOp<s_N>::NBackwardOp;
};

class BackwardUnsqueeze : public BackwardView {
public:
    size_t m_dim;
    
    BackwardUnsqueeze(
        Tensor viewed_tensor,
        const size_t dim
    );

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardSqueeze : public BackwardView {
public:
    size_t m_dim;
    
    BackwardSqueeze(
        Tensor viewed_tensor,
        const size_t dim
    );

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardRepeat : public BackwardView {
public:
    const size_t m_dim;
    
    BackwardRepeat(
        Tensor viewed_tensor,
        const size_t dim
    );

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

class BackwardClone : public BackwardView {
public:
    size_t m_dim;
    
    BackwardClone(
        Tensor viewed_tensor
    );

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(const Tensor& out, bool create_graph = false) override;
};

template <size_t N>
class NBackwardComposite : public NBackwardOp<N> {
public:
    using NBackwardOp<N>::m_operands;
    Tensor m_internal_head;

    template <typename... Tensors>
    explicit NBackwardComposite(
        Tensor internal_head,
        Tensors... operands
    ): 
        NBackwardOp<N>(operands...),
        m_internal_head(internal_head) {}
    
    void compute_operands_grad(
        const Tensor& out,
        bool create_graph
    ) {
        auto original_operands_bw_ops = save_and_detach_operands_bw_ops();

        m_internal_head.accumulate_grad(out.grad(), create_graph);
        
        std::map<TensorNode*, int> in_degree { m_internal_head.compute_in_degree() };
        m_internal_head.topological_backprop(in_degree, create_graph);

        restore_operands_bw_ops(std::move(original_operands_bw_ops));
    }

private:
    std::array<std::unique_ptr<BackwardOp>, N> save_and_detach_operands_bw_ops() {
        std::array<std::unique_ptr<BackwardOp>, N> original_operands_bw_ops;
        for (size_t i {0}; i < m_operands.size(); i++) {
            original_operands_bw_ops[i] = std::move(m_operands[i].m_node->m_bw_op);
            m_operands[i].detach();
        }
        return original_operands_bw_ops;
    }

    void restore_operands_bw_ops(
        std::array<std::unique_ptr<BackwardOp>, N> original_operands_bw_ops
    ) {
        for (size_t i {0}; i < m_operands.size(); i++) {
            m_operands[i].m_node->m_bw_op = std::move(original_operands_bw_ops[i]);
        }
    }
};

class BackwardMatMul : public NBackwardComposite<2> {
public:
    using NBackwardComposite<s_N>::NBackwardComposite;

    std::ostream& print(std::ostream& os) const override;
};

#endif
