#ifndef TENSORS_H
#define TENSORS_H

#include <iostream>
#include <vector>
#include <mdspan>
#include <iomanip>
#include <string>
#include <memory>

#include "ops.h"

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector){
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
    size_t m_offset;
    
    std::shared_ptr<Tensor> m_grad;
    std::shared_ptr<BackwardOp> m_grad_fn;

    Tensor(
        std::vector<size_t> shape
    );
    
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    
    size_t get_flat_index_from_md(
        const std::vector<size_t>& md_index
    ) const;

    size_t get_flat_index_from_logical(
        size_t logical_index
    ) const;

    bool is_contiguous();
    
    float& operator[](const std::vector<size_t>& md_index);
    
    void fill(
        float value
    );
    
    float& item();
    
    void slice(
        size_t dim,
        size_t slice_index
    );
    
    void dice(
        size_t dim,
        size_t index_start,
        size_t index_end
    );
    
    void linspace(
        float start,
        float end
    );
    
    void reshape(
        std::vector<size_t> shape
    );

    void transpose(
        size_t dim_1,
        size_t dim_2
    );

    static bool are_shapes_equal(
        Tensor& a,
        Tensor& b
    );

    Tensor operator*(
        Tensor& other
    );
    
    Tensor operator+(
        Tensor& other
    );
    
    Tensor operator-();
    
    Tensor pow(
        Tensor& exp
    );

    void reset_grads();

    void backward();

    void backprop();
    
private:
    static std::vector<size_t> init_strides(
        const std::vector<size_t>& shape
    );

    float& get_entry_ref(
        size_t l_index
    );

    static Tensor mult(
        Tensor& a,
        Tensor& b
    );  
};

#endif
