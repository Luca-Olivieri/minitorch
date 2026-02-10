#ifndef TENSORS_H
#define TENSORS_H

#include <memory>
#include <vector>
#include <iostream>

class TensorNode;

class Tensor {
public:
    std::shared_ptr<TensorNode> m_node;

    friend class TensorNode;

    Tensor(
        std::vector<size_t> shape
    );
    
    Tensor(
        std::shared_ptr<TensorNode> node
    );

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    
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
    
    Tensor grad() const;

    void zero_grad();

    const std::vector<size_t>& shape();

    void backward(
        bool create_graph = false
    );

    void accumulate_grad(
        const Tensor& gradient,
        bool create_graph = false
    );

private:
};


#endif
