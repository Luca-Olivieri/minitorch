#include "compute.h"
#include "src/core/reproducibility.h"
#include "src/core/formatting.h"
#include <cmath>
#include <string>
#include <stdexcept>

namespace mt::nn {
    
    Linear::Linear(
        const size_t in_features,
        const size_t out_features,
        const bool bias
    ) {
        // Xavier/Glorot uniform initialization to break symmetry between units
        m_weight = Tensor({in_features, out_features}, 0.0f, true);
        m_parameters.emplace("weight", m_weight);
    
        xavier_uniform_inplace(m_weight, get_rng());
    
        if (bias) {
            m_bias = Tensor({out_features}, 0.0f, true);
            m_parameters.emplace("bias", m_bias);
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
            std::mt19937&& rng
    ) {
        // limit = sqrt(6 / (in + out))
        const float limit = std::sqrt(6.0f / static_cast<float>(x.shape()[0] + x.shape()[1]));
        std::uniform_real_distribution<float> dist(-limit, limit);
        
        for (size_t i {0}; i < x.numel(); ++i) {
            x.m_node->m_storage.get_entry_ref(i) = dist(rng);
        }
    }

    void xavier_normal_inplace(
            Tensor& x,
            std::mt19937&& rng
    ) {
        // stddev = sqrt(2 / (in + out)) to match variance of uniform version
        const float stddev = std::sqrt(2.0f / static_cast<float>(x.shape()[0] + x.shape()[1]));
        std::normal_distribution<float> dist(0.0f, stddev);

        for (size_t i {0}; i < x.numel(); ++i) {
            x.m_node->m_storage.get_entry_ref(i) = dist(rng);
        }
    }

    float calculate_gain(
            const std::string& nonlinearity,
            float param
    ) {
        if (nonlinearity == "tanh") return 5.0f / 3.0f;
        if (nonlinearity == "relu") return std::sqrt(2.0f);
        if (nonlinearity == "leaky_relu") return std::sqrt(2.0f / (1.0f + param * param));
        return 1.0f;
    }

    void kaiming_uniform_inplace(
            Tensor& x,
            std::mt19937&& rng,
            const std::string& fan_mode,
            const std::string& nonlinearity,
            float param
    ) {
        const float gain = calculate_gain(nonlinearity, param);
        size_t fan = 0;
        if (fan_mode == "fan_in" || fan_mode == "in") {
            fan = x.shape()[0];
        } else if (fan_mode == "fan_out" || fan_mode == "out") {
            fan = x.shape()[1];
        } else {
            throw std::invalid_argument("kaiming_uniform_inplace: invalid fan_mode; expected 'fan_in' or 'fan_out'");
        }

        const float stddev = (fan > 0) ? gain / std::sqrt(static_cast<float>(fan)) : 0.0f;
        const float bound = std::sqrt(3.0f) * stddev;
        std::uniform_real_distribution<float> dist(-bound, bound);

        for (size_t i {0}; i < x.numel(); ++i) {
            x.m_node->m_storage.get_entry_ref(i) = dist(rng);
        }
    }

    void kaiming_normal_inplace(
            Tensor& x,
            std::mt19937&& rng,
            const std::string& fan_mode,
            const std::string& nonlinearity,
            float param
    ) {
        const float gain = calculate_gain(nonlinearity, param);
        size_t fan = 0;
        if (fan_mode == "fan_in" || fan_mode == "in") {
            fan = x.shape()[0];
        } else if (fan_mode == "fan_out" || fan_mode == "out") {
            fan = x.shape()[1];
        } else {
            throw std::invalid_argument("kaiming_normal_inplace: invalid fan_mode; expected 'fan_in' or 'fan_out'");
        }

        const float stddev = (fan > 0) ? gain / std::sqrt(static_cast<float>(fan)) : 0.0f;
        std::normal_distribution<float> dist(0.0f, stddev);

        for (size_t i {0}; i < x.numel(); ++i) {
            x.m_node->m_storage.get_entry_ref(i) = dist(rng);
        }
    }
}
