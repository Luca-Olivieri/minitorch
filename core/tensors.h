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
    size_t m_size;
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
    
    template<typename... Indices>
    float operator[](Indices... indices) {
        // Logic to map N indices to 1 flat index
        // std::vector<size_t> md_index = { static_cast<size_t>(indices)... };
        std::vector<size_t> md_index = {static_cast<size_t>(indices)...};
        return m_flat_data[get_flat_index(md_index)];
    }

    void fill(
        float value
    );
private:
    static std::vector<size_t> init_strides(
        const std::vector<size_t>& shape
    );

    // template<typename... Args>
    // float operator[](Args... indices) {
    //     // Logic to map N indices to 1 flat index
    //     return data[]; 
    // }
    
};

#endif
