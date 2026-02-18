#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "modules.h"

namespace mt::nn {
    class ReLU : public Module, public Forward1 {
    public:
        ReLU();

        Tensor forward(
            const Tensor& input
        ) const override;
    };
}

#endif
