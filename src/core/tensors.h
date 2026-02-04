#ifndef TENSORS_H
#define TENSORS_H

#include "tensors_impl.h"

class BackwardOp;

class Tensor {
public:
    TensorImpl m_value;
    
    std::shared_ptr<BackwardOp> m_bw_op { nullptr };
    std::shared_ptr<Tensor> m_grad { nullptr };
    bool m_requires_grad { true };

    Tensor(
        std::vector<size_t> shape
    );

    // TODO provide a constructor which accepts a TensorImpl

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    
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
    Tensor apply_op(
        Tensors&... others
    ) const {
        // 1. Infer N at compile time
        constexpr size_t N = sizeof...(others) + 1;
        static_assert(N == BW_OP::N, "Number of tensors does not match BackwardOp arity");

        Tensor out{m_value.m_shape};
        out.m_value = Op(m_value, others.m_value...);
        out.m_bw_op = std::make_shared<BW_OP>(
            std::array<Tensor*, BW_OP::N>(
                    {
                        const_cast<Tensor*>(this),
                        const_cast<Tensor*>(&others)...
                    }
                )
            );

        return out;
    }
    
    Tensor operator+(
        const Tensor& other
    ) const;
    
    void operator+=(
        const Tensor& other
    );
    
    Tensor operator-() const;
    
    Tensor operator*(
        const Tensor& other
    ) const;
    
    Tensor pow(
        const Tensor& other
    ) const;
};

#endif
