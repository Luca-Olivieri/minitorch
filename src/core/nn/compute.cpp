#include "compute.h"
#include "src/core/formatting.h"
#include <random>

using namespace mt::nn;

Linear::Linear(
    const size_t in_features,
    const size_t out_features,
    const bool bias

) {
    // Xavier/Glorot uniform initialization to break symmetry between units
    m_weight = Tensor({in_features, out_features}, 0.0f, true);

    // limit = sqrt(6 / (in + out))
    const float limit = std::sqrt(6.0f / static_cast<float>(in_features + out_features));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    const auto w_shape = m_weight.shape();
    for (size_t i = 0; i < w_shape[0]; ++i) {
        for (size_t j = 0; j < w_shape[1]; ++j) {
            m_weight[{i, j}] = dist(gen);
        }
    }

    if (bias) {
        m_bias = Tensor({out_features}, 0.0f, true);
        const auto b_shape = m_bias.shape();
        for (size_t i = 0; i < b_shape[0]; ++i) {
            m_bias[{i}] = dist(gen);
        }
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
