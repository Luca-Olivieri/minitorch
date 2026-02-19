#include "activations.h"
#include "src/core/tensors.h"
#include "src/core/grad_fns.h"

namespace mt::nn {
    
    ReLU::ReLU() {}
    
    Tensor ReLU::forward(
        const Tensor& input
    ) const {
        Tensor zeros(input.shape(), 0.0f, false);
        return Tensor::maximum(input, zeros); // dy/dx = 0 when x = 0
    }
}
