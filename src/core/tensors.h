#ifndef TENSORS_H
#define TENSORS_H

#include "tensor_nodes.h"

class Tensor {
public:
    std::shared_ptr<TensorNode> m_node;

    Tensor(
        std::vector<size_t> shape
    );

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
        Tensor& other
    );
    
    void operator+=(
        const Tensor& other
    );

    Tensor operator-();
    
    Tensor operator*(
        Tensor& other
    );
    
    Tensor pow(
        Tensor& other
    );
    
    void zero_grad();

    void backward();

    void backprop();

private:
    
    Tensor();
    
    Tensor(
        std::shared_ptr<TensorNode> node
    );
};


#endif
