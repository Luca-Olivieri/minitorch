#include "compute.h"
#include "src/core/formatting.h"
#include <random>

namespace mt::nn {
    
    Linear::Linear(
        const size_t in_features,
        const size_t out_features,
        const bool bias
    
    ) {
        // Xavier/Glorot uniform initialization to break symmetry between units
        m_weight = Tensor({in_features, out_features}, 0.0f, true);
    
        xavier_uniform_inplace(m_weight, in_features, out_features);
    
        if (bias) {
            m_bias = Tensor({out_features}, 0.0f, true);
        }
    }
    
    Tensor Linear::forward(
        const Tensor& inputs
    ) const {
        Tensor mult = Tensor::matmul(inputs, m_weight);
        if (inputs.shape().size() == 1) {
            return mult + m_bias;
        }
        else {
            return mult + m_bias.unsqueeze(0).repeat(0, inputs.shape()[0]);
        }    
    }
    
    void xavier_uniform_inplace(
            Tensor& x,
            const size_t in_features,
            const size_t out_features
    ) {
        // limit = sqrt(6 / (in + out))
        const float limit = std::sqrt(6.0f / static_cast<float>(in_features + out_features));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-limit, limit);
    
        const auto w_shape = x.shape();
        for (size_t i {0}; i < x.numel(); ++i) {
            x.m_node->m_storage.get_entry_ref(i) = dist(gen);
        }
    }
}
