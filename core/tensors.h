#ifndef TENSORS_H
#define TENSORS_H

#include <iostream>
#include <vector>
#include <mdspan>
#include <iomanip>
#include <string>

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector){
    std::string out_string = "[";
    for (size_t i { 0 }; i < vector.size(); i++) {
        out_string += std::to_string(vector[i]);
        if (i != vector.size()-1) {
            out_string += ", ";
        }
    }
    out_string += "]";
    return os << out_string;
}

class Tensor {
public:
    std::vector<size_t> m_shape;
    size_t m_numel;
    std::vector<size_t> m_strides;
    std::vector<float> m_flat_data;

    Tensor(
        std::initializer_list<size_t> shape
    );

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    size_t get_flat_index(
        const std::vector<size_t>& md_index
    );
    
    template<typename... Indices>
    float operator[](Indices... indices) {
        // Convert variadic indices into a std::vector
        std::vector<size_t> md_index = {static_cast<size_t>(indices)...};
        // Scalar case
        if (m_shape.empty()) {
            throw std::runtime_error(std::format("\nScalar tensor of shape cannot be access by index. Got index {}", md_index));
        }
        // Bounds check
        if (md_index.size() > m_shape.size()) {
            throw std::runtime_error(std::format("\nTensor of shape {} accessed at larger index {}", m_shape, md_index));
        }
        if (md_index.size() < m_shape.size()) {
            throw std::runtime_error(std::format("\nTensor of shape {} accessed at partial index {}", m_shape, md_index));
        }
        for (size_t i { 0 }; i < m_shape.size(); i++) {
            if (md_index[i] >= m_shape[i]) {
                throw std::out_of_range(std::format("\nTensor of shape {} accessed out-of-bounds at index {}", m_shape, md_index));
            }
            if (md_index[i] < 0) {
                // TODO: right now, indices cannot be negative because size_t rolls up.
                throw std::out_of_range(std::format("\nTensor accessed at negative index {}", md_index));
            }
        }
        float flat_index = m_flat_data[get_flat_index(md_index)];
        return flat_index;
    }

    void fill(
        float value
    );

    float item();
private:
    static std::vector<size_t> init_strides(
        const std::vector<size_t>& shape
    );    
};

#endif
