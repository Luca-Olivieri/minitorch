#ifndef TEST_MSELOSS_H
#define TEST_MSELOSS_H

#include "src/core/tensors.h"
#include "src/core/nn/losses.h"
#include "tests/test_utils.h"

void test_mse_loss_forward_backward() {
    std::cout << "\n===[ test_nn/losses: MSELoss ]===\n";

    // Simple 1D example
    Tensor inputs({3});
    inputs[{0}] = 1.0f;
    inputs[{1}] = 2.0f;
    inputs[{2}] = 3.0f;

    // targets set to 2.0 for all elements, do not require grad
    Tensor targets({3}, 2.0f, false);

    mt::nn::MSELoss loss;
    Tensor out = loss.forward(inputs, targets);

    // Forward: ((2-1)^2 + (2-2)^2 + (2-3)^2) / 3 = (1 + 0 + 1) / 3 = 2/3
    ASSERT_EQ_APPROX(out.item(), 2.0f/3.0f, 1e-6, "MSE forward value");

    // Backward: d/dx_i = (2/N) * (x_i - t_i)
    out.backward();

    Tensor g = inputs.grad();
    ASSERT_EQ_APPROX(g[{0}], (2.0f/3.0f) * (1.0f - 2.0f), 1e-6, "grad[0] == 2/3 * (x0 - t0)");
    ASSERT_EQ_APPROX(g[{1}], (2.0f/3.0f) * (2.0f - 2.0f), 1e-6, "grad[1] == 0");
    ASSERT_EQ_APPROX(g[{2}], (2.0f/3.0f) * (3.0f - 2.0f), 1e-6, "grad[2] == 2/3 * (x2 - t2)");
}

#endif
