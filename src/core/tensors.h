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

    Tensor operator+(
        const Tensor& other
    ) const;
};

#endif
