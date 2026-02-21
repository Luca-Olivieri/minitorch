#ifndef TEST_SOFTMAX_H
#define TEST_SOFTMAX_H

#include "src/core/tensors.h"
#include "src/core/nn/activations.h"
#include "tests/test_utils.h"

void test_softmax_forward_backward() {
    std::cout << "\n===[ test_Softmax.h ]===\n";

    // Forward: simple vector
    {
        Tensor t({3});
        t[{0}] = 0.0f;
        t[{1}] = 1.0f;
        t[{2}] = 2.0f;

        mt::nn::Softmax sm;
        Tensor out = sm.forward(t);

        const float e = std::exp(1.0f);
        const float s0 = 1.0f;
        const float s1 = e;
        const float s2 = e * e;
        const float denom = s0 + s1 + s2;

        ASSERT_EQ_APPROX(out[{0}], s0 / denom, 1e-6, "Softmax forward value 0");
        ASSERT_EQ_APPROX(out[{1}], s1 / denom, 1e-6, "Softmax forward value 1");
        ASSERT_EQ_APPROX(out[{2}], s2 / denom, 1e-6, "Softmax forward value 2");
    }

    // Forward: batch (2x3) row-wise softmax -> rows sum to 1 and match elementwise
    {
        Tensor t({2,3});
        // row 0: [0,1,2]
        t[{0,0}] = 0.0f; t[{0,1}] = 1.0f; t[{0,2}] = 2.0f;
        // row 1: [2,1,0]
        t[{1,0}] = 2.0f; t[{1,1}] = 1.0f; t[{1,2}] = 0.0f;

        mt::nn::Softmax sm;
        Tensor out = sm.forward(t);

        // each row should sum to 1
        Tensor row0_sum = out.sum(1).unsqueeze(1); // sums over last dim
        ASSERT_EQ_APPROX(row0_sum[{0,0}], 1.0f, 1e-6, "Softmax row0 sums to 1");
        ASSERT_EQ_APPROX(row0_sum[{1,0}], 1.0f, 1e-6, "Softmax row1 sums to 1");
    }

    // Backward: when upstream grad is ones, gradient w.r.t inputs should be zero (property of softmax)
    {
        Tensor t({2});
        t[{0}] = 0.5f;
        t[{1}] = -1.0f;

        mt::nn::Softmax sm;
        Tensor out = sm.forward(t);

        out.backward();

        Tensor g = t.grad();
        ASSERT_EQ_APPROX(g[{0}], 0.0f, 1e-6, "Softmax backward upstream ones -> grad 0 (elem0)");
        ASSERT_EQ_APPROX(g[{1}], 0.0f, 1e-6, "Softmax backward upstream ones -> grad 0 (elem1)");
    }
}

#endif
