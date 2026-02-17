#ifndef BACKWARD_OPS
#define BACKWARD_OPS

#include <iostream>
#include <memory>
#include <vector>
#include <functional>

#include "tensor_nodes.h"
#include "tensors.h"

class GradFn {
public:
    GradFn();

    virtual ~GradFn() = default;

    virtual std::ostream& print(std::ostream& os) const = 0;
    friend std::ostream& operator<<(std::ostream& os, const GradFn& op);
    
    virtual void reset_all_grads() = 0;
    
    virtual void compute_operands_grad(
        const Tensor& out
    ) = 0;

    virtual std::vector<Tensor> get_operands() const = 0;
};

template <size_t N>
class NBackwardOp: public GradFn {
public:
    static constexpr size_t s_N = N; // expose N as a static member
    std::array<Tensor, N> m_operands;

        // 1. Variadic Template Constructor
    template <typename... Tensors>
    explicit NBackwardOp(
            const Tensors... operands
    ): m_operands{{operands...}} { // Optimization note: Move semantics (Tensors&&... and std::forward) could be used here for efficiency to avoid atomic ref-count increments on shared_ptr
        // 2. Compile-time Arity Check
        static_assert(sizeof...(Tensors) == N, 
            "Error: Number of arguments provided to constructor must match template parameter N.");
            
        // 3. (Optional) Strict Type Checking using C++17 Fold Expressions
        // This ensures every argument is actually a Tensor
        static_assert((std::is_convertible_v<Tensors, Tensor> && ...), 
            "Error: All arguments must be implicitly convertible to Tensor.");
    }

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
    
    void compute_operands_grad(
            const Tensor& out
    ) override;
};

class BackwardMinus : public NBackwardOp<1> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(
            const Tensor& out
    ) override;
};

class BackwardSub : public NBackwardOp<2> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(
            const Tensor& out
    ) override;
};

class BackwardMult : public NBackwardOp<2> {
public:    
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(
            const Tensor& out
    ) override;
};

class BackwardPow : public NBackwardOp<2> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(
            const Tensor& out
    ) override;
};

class BackwardDiv : public NBackwardOp<2> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(
            const Tensor& out
    ) override;
};

class BackwardLog : public NBackwardOp<1> {
public:
    using NBackwardOp<s_N>::NBackwardOp;

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(
            const Tensor& out
    ) override;
};

class BackwardReduce : public NBackwardOp<1> {
public:
    using NBackwardOp<s_N>::NBackwardOp;
};

class BackwardSum : public BackwardReduce {
public:
    const size_t m_dim;
    const size_t m_original_times;
    
    BackwardSum(
            const Tensor reduced_tensor,
            const size_t dim,
            const size_t original_times
    );

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(
            const Tensor& out
    ) override;
};

class BackwardView : public NBackwardOp<1> {
public:
    using NBackwardOp<s_N>::NBackwardOp;
};

class BackwardUnsqueeze : public BackwardView {
public:
    const size_t m_dim;
    
    BackwardUnsqueeze(
            const Tensor viewed_tensor,
            const size_t dim
    );

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(
            const Tensor& out
    ) override;
};

class BackwardSqueeze : public BackwardView {
public:
    const size_t m_dim;
    
    BackwardSqueeze(
            const Tensor viewed_tensor,
            const size_t dim
    );

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(
            const Tensor& out
    ) override;
};

class BackwardRepeat : public BackwardView {
public:
    const size_t m_dim;
    
    BackwardRepeat(
            const Tensor viewed_tensor,
            const size_t dim
    );

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(
            const Tensor& out
    ) override;
};

class BackwardClone : public BackwardView {
public:
    size_t m_dim;
    
    BackwardClone(
        const Tensor viewed_tensor
    );

    std::ostream& print(std::ostream& os) const override;
    
    void compute_operands_grad(
            const Tensor& out
    ) override;
};

#endif
