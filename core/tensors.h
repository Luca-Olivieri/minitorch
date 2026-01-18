#ifndef TENSORS_H
#define TENSORS_H

#include <iostream>
#include <vector>
#include <mdspan>
#include <iomanip>

std::ostream& operator<<(std::ostream& os, const std::vector<float>& vector);

class Tensor {
public:
    std::vector<size_t> m_shape;
    std::vector<size_t> m_strides;
    std::vector<float> m_flat_data;

    Tensor(
        std::initializer_list<size_t> shape
    );

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    // C++23 Multi-argument subscript operator
    // float operator[](int x) {
    //     return m_flat_data[0];
    // }

    size_t get_flat_index(
        const std::vector<size_t>& md_index
    );
private:
    static std::vector<float> init_flat_data(
        const std::vector<size_t>& shape
    );
    
    static std::vector<size_t> init_strides(
        const std::vector<size_t>& shape
    );
    
};

#endif
