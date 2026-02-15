#ifndef TEST_SQUEEZE_H
#define TEST_SQUEEZE_H

#include "src/core/tensors.h"
#include "tests/test_utils.h"

void test_squeeze() {
    std::cout << "\n===[ test_squeeze.h ]===\n";

    // 1. Squeeze a leading singleton
    {
        Tensor t = Tensor::linspace({1, 2, 3}, 0.0f, 5.0f);

        Tensor s = t.squeeze(0); // shape -> {2,3}

        ASSERT_EQ(s.shape().size(), static_cast<size_t>(2), "squeezed rank is 2");
        ASSERT_EQ(s.shape()[0], static_cast<size_t>(2), "squeezed dim 0 == 2");
        ASSERT_EQ(s.shape()[1], static_cast<size_t>(3), "squeezed dim 1 == 3");

        ASSERT_EQ(s[{0, 0}], t[{0, 0, 0}], "value preserved after squeeze at 0,0");
        ASSERT_EQ(s[{1, 2}], t[{0, 1, 2}], "value preserved after squeeze at 1,2");
    }

    // 2. Squeeze out-of-range dimension should throw
    {
        Tensor t({1, 2, 3});
        ASSERT_THROWS(t.squeeze(3), std::invalid_argument);
    }

    // 3. Backward: gradient should flow through squeeze to original tensor
    {
        Tensor t = Tensor::linspace({1, 2, 3}, 0.0f, 5.0f); // shape {1,2,3}

        Tensor s = t.squeeze(0); // shape {2,3}

        // Run backward on the squeezed view
        s.backward();

        Tensor g = t.grad();

        ASSERT_EQ(g.shape().size(), static_cast<size_t>(3), "grad rank for original after squeeze backward is 3");
        ASSERT_EQ(g.shape()[0], static_cast<size_t>(1), "grad dim0 == 1");
        ASSERT_EQ(g.shape()[1], static_cast<size_t>(2), "grad dim1 == 2");
        ASSERT_EQ(g.shape()[2], static_cast<size_t>(3), "grad dim2 == 3");

        // All gradients should be ones (since out.grad is ones)
        ASSERT_EQ(g[{0,0,0}], 1.0f, "grad preserved at 0,0,0 after squeeze backward");
        ASSERT_EQ(g[{0,1,2}], 1.0f, "grad preserved at 0,1,2 after squeeze backward");
    }
}

#endif
