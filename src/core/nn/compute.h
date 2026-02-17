#ifndef COMPUTE_H
#define COMPUTE_H

#include "modules.h"

namespace mt::nn {
    class Linear : public Module {
    public:
        Tensor m_weight = Tensor(nullptr);
        Tensor m_bias = Tensor(nullptr);

        Linear(
            const size_t in_features,
            const size_t out_features,
            const bool bias = true
        );

        Tensor forward(
            const Tensor& input
        ) const override;
    };
}


#endif
