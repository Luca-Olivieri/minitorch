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
    
    float& operator[](const std::vector<size_t>& md_index);

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
