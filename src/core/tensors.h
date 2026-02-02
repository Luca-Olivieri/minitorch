#ifndef TENSORS_H
#define TENSORS_H

#include "tensors_impl.h"

class Tensor {
public:
    TensorImpl m_value;

    
    Tensor(
        std::vector<size_t> shape
    );
    
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

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
};

#endif
