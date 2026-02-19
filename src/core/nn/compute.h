#ifndef COMPUTE_H
#define COMPUTE_H

#include "modules.h"
#include <random>
#include <string>

namespace mt::nn {
    class Linear : public Module, public Forward1 {
    public:
        Tensor m_weight = Tensor(nullptr);
        Tensor m_bias = Tensor(nullptr);

        Linear(
                const size_t in_features,
                const size_t out_features,
                const bool bias = true
        );

        Tensor forward(
                const Tensor& inputs
        ) const override;
    };

    void xavier_uniform_inplace(
            Tensor& x,
            std::mt19937&& rng
    );
    
    void xavier_normal_inplace(
            Tensor& x,
            std::mt19937&& rng
    );

    float calculate_gain(
            const std::string& nonlinearity,
            float param = 0.0f
    );

    void kaiming_uniform_inplace(
            Tensor& x,
            std::mt19937&& rng,
            const std::string& fan_mode = "fan_in",
            const std::string& nonlinearity = "relu",
            float param = 0.0f
    );
    
    void kaiming_normal_inplace(
            Tensor& x,
            std::mt19937&& rng,
            const std::string& fan_mode = "fan_in",
            const std::string& nonlinearity = "relu",
            float param = 0.0f
    );
}


#endif
