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
    
    Softmax::Softmax() {}
    
    Tensor Softmax::forward(
        const Tensor& input
    ) const {
        const size_t ndim = input.shape().size();
        if (ndim == 0) {
            Tensor ones(input.shape(), 1.0f, false);
            return ones;
        }

        const size_t dim = ndim - 1; // softmax over the last dimension

        // compute exponentials using e^x via elementwise pow with base e
        Tensor e_const(input.shape(), std::exp(1.0f), false);
        Tensor exps = e_const.pow(input);

        // sum over the target dimension and broadcast for division
        Tensor sums = exps.sum(dim);
        Tensor denom = sums.unsqueeze(dim).expand(dim, input.shape()[dim]);

        return exps / denom;
    }
}
