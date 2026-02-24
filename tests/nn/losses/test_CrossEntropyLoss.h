#ifndef TEST_CROSSENTROPYLOSS_H
#define TEST_CROSSENTROPYLOSS_H

#include "src/core/tensors.h"
#include "src/core/nn/losses.h"
#include "tests/test_utils.h"

void test_crossentropy_loss_forward_backward() {
    std::cout << "\n===[ test_nn/losses: CrossEntropyLoss ]===\n";

    // 2 samples, 3 classes
    Tensor inputs({2,3});
    // sample 0 logits
    inputs[{0,0}] = 1.0f; inputs[{0,1}] = 2.0f; inputs[{0,2}] = 3.0f;
    // sample 1 logits
    inputs[{1,0}] = 0.5f; inputs[{1,1}] = -1.0f; inputs[{1,2}] = 0.0f;

    // one-hot targets
    Tensor targets({2,3});
    targets[{0,2}] = 1.0f; // class 2 for sample 0
    targets[{1,0}] = 1.0f; // class 0 for sample 1
    // mark targets non-differentiable
    targets.detach_inplace();

    mt::nn::CrossEntropyLoss loss;
    Tensor out = loss.forward(inputs, targets);
    // std::cout << "[DEBUG] forward returned, out dims=" << out.shape().size() << "\n";

    // compute expected using softmax + negative log-likelihood manually
    // sample 0 softmax probs: exp([1,2,3]) / sum = [e1,e2,e3]/(e1+e2+e3)
    Tensor e_const(inputs.shape(), std::exp(1.0f), false);
    Tensor exps = e_const.pow(inputs);
    Tensor sums = exps.sum(1);
    Tensor denom = sums.unsqueeze(1).expand(1, inputs.shape()[1]);
    Tensor probs = exps / denom;

    // NLL per sample: -sum(target * log(probs), dim=1)
    Tensor logp = probs.log();
    Tensor expected = -(targets * logp).sum(1);
    // std::cout << "[DEBUG] expected computed, dims=" << expected.shape().size() << "\n";

    Tensor expected_mean = expected.mean(0);
    // std::cout << "[DEBUG] expected mean dims=" << expected_mean.shape().size() << "\n";
    ASSERT_EQ_APPROX(out.item(), expected_mean.item(), 1e-6, "crossentropy forward mean");

    // Backward: compute grads w.r.t. inputs by calling backward()
    out.backward();
    // std::cout << "[DEBUG] backward called\n";
    Tensor g = inputs.grad();
    // std::cout << "[DEBUG] grad obtained, dims=" << g.shape().size() << "\n";

    // For cross-entropy with softmax and one-hot targets, grad = probs - targets, averaged over batch
    // Since loss.mean(0) was used, check that gradient matches (probs - targets) / N
    const float invN = 1.0f / static_cast<float>(inputs.shape()[0]);
    for (size_t i = 0; i < inputs.shape()[0]; ++i) {
        for (size_t j = 0; j < inputs.shape()[1]; ++j) {
            float expected_grad = (probs[{i,j}] - targets[{i,j}]) * invN;
            ASSERT_EQ_APPROX(g[{i,j}], expected_grad, 1e-6, "grad matches softmax-crossentropy derivative");
        }
    }
}

#endif
