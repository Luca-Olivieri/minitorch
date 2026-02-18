#include "losses.h"
#include "src/core/formatting.h"

using namespace mt::nn;

MSELoss::MSELoss() {}

Tensor MSELoss::forward(
        const Tensor& inputs,
        const Tensor& targets
) const {
    Tensor diff(targets - inputs);
    // Tensor twos(diff.shape(), 2.0f, false);
    // Tensor squared = diff.pow(twos);
    Tensor squared = diff * diff;
    return squared.mean(0);
}
