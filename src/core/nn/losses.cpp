#include "losses.h"
#include "activations.h"
#include "src/core/formatting.h"
#include <cmath>

namespace mt::nn {

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
    
    BCELossWithLogits::BCELossWithLogits() {}
    
    Tensor BCELossWithLogits::forward(
            const Tensor& inputs,
            const Tensor& targets
    ) const {
        // BCE with logits stable formulation:
        // loss = max(x, 0) - x * y + log(1 + exp(-abs(x)))
        Tensor zeros(inputs.shape(), 0.0f, false);
        Tensor max_val = Tensor::maximum(inputs, zeros);
        Tensor abs_inputs = Tensor::maximum(inputs, -inputs);
    
        Tensor e_const(inputs.shape(), std::exp(1.0f), false);
        Tensor exp_term = e_const.pow(-abs_inputs);
        Tensor ones(inputs.shape(), 1.0f, false);
        Tensor log_term = (exp_term + ones).log();
    
        Tensor loss = max_val - (inputs * targets) + log_term;
        return loss.mean(0);
    }
    
    CrossEntropyLoss::CrossEntropyLoss() {}
    
    Tensor CrossEntropyLoss::forward(
            const Tensor& inputs,
            const Tensor& targets
    ) const {
        const size_t ndim = inputs.shape().size();
        if (ndim == 0) {
            Tensor probs = Softmax().forward(inputs);
            Tensor logp = probs.log();
            Tensor loss = -(targets * logp);
            return loss.mean(0);
        }

        const size_t dim = ndim - 1; // softmax over last dimension

        // compute probabilities via softmax
        Tensor probs = Softmax().forward(inputs);

        // log probabilities
        Tensor log_probs = probs.log();

        // elementwise multiply with targets (expects one-hot targets)
        Tensor mul = targets * log_probs;

        // sum over class dimension and take negative
        Tensor summed = mul.sum(dim);
        Tensor loss = -summed;

        return loss.mean(0);
    }
}
