#ifndef TEST_ACTIVATIONS_H
#define TEST_ACTIVATIONS_H

#include "src/core/tensors.h"
#include "src/core/nn/activations.h"
#include "tests/test_utils.h"

void test_relu_forward_backward() {
    std::cout << "\n===[ test_activations.h ]===\n";

    // Forward: elementwise max(0, x)
    {
        Tensor t({3});
        t[{0}] = -1.0f;
        t[{1}] = 0.0f;
        t[{2}] = 2.5f;

        mt::nn::ReLU relu;
        Tensor out = relu.forward(t);

        ASSERT_EQ(out[{0}], 0.0f, "ReLU forward negative -> 0");
        ASSERT_EQ(out[{1}], 0.0f, "ReLU forward zero -> 0");
        ASSERT_EQ(out[{2}], 2.5f, "ReLU forward positive unchanged");
    }

    // Backward: gradient flows only where input > 0
    {
        Tensor t({3});
        t[{0}] = -2.0f; // should get zero grad
        t[{1}] = 0.0f;  // borderline: current impl treats zero as non-positive -> zero grad
        t[{2}] = 1.0f;  // should get grad 1

        mt::nn::ReLU relu;
        Tensor out = relu.forward(t);

        out.backward();

        Tensor g = t.grad();
        ASSERT_EQ(g[{0}], 0.0f, "ReLU backward negative -> grad 0");
        ASSERT_EQ(g[{1}], 0.0f, "ReLU backward zero -> grad 0 (implementation choice)");
        ASSERT_EQ(g[{2}], 1.0f, "ReLU backward positive -> grad 1");
    }
}

#endif
