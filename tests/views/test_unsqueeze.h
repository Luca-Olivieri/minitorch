#ifndef TEST_UNSQUEEZE_H
#define TEST_UNSQUEEZE_H

#include "src/core/tensors.h"
#include "tests/test_utils.h"

void test_unsqueeze() {
    std::cout << "\n===[ test_unsqueeze.h ]===\n";

    // 1. Unsqueeze at front (dim 0)
    {
        Tensor t({2, 3});
        // fill with distinct values
        t[{0, 0}] = 1.0f; t[{0, 1}] = 2.0f; t[{0, 2}] = 3.0f;
        t[{1, 0}] = 4.0f; t[{1, 1}] = 5.0f; t[{1, 2}] = 6.0f;

        Tensor u = t.unsqueeze(0);

        ASSERT_EQ(u.shape().size(), static_cast<size_t>(3), "unsqueezed rank is 3");
        ASSERT_EQ(u.shape()[0], static_cast<size_t>(1), "unsqueezed dim 0 == 1");
        ASSERT_EQ(u.shape()[1], static_cast<size_t>(2), "unsqueezed dim 1 == 2");
        ASSERT_EQ(u.shape()[2], static_cast<size_t>(3), "unsqueezed dim 2 == 3");

        ASSERT_EQ(u[{0, 0, 0}], 1.0f, "value preserved at 0,0,0");
        ASSERT_EQ(u[{0, 1, 2}], 6.0f, "value preserved at 0,1,2");
    }

    // 2. Unsqueeze at end (append singleton)
    {
        Tensor t({2, 3});
        t.fill_inplace(7.0f);

        Tensor u = t.unsqueeze(2); // shape -> {2,3,1}

        ASSERT_EQ(u.shape().size(), static_cast<size_t>(3), "unsqueezed rank is 3 (append)");
        ASSERT_EQ(u.shape()[0], static_cast<size_t>(2), "unsqueezed dim 0 == 2");
        ASSERT_EQ(u.shape()[1], static_cast<size_t>(3), "unsqueezed dim 1 == 3");
        ASSERT_EQ(u.shape()[2], static_cast<size_t>(1), "unsqueezed dim 2 == 1");

        ASSERT_EQ(u[{1, 2, 0}], 7.0f, "value preserved at appended singleton index");
    }

    // 3. Out of range unsqueeze should throw
    {
        Tensor t({2, 2});
        ASSERT_THROWS(t.unsqueeze(3), std::invalid_argument);
    }

    // 4. Backward: gradient should flow through unsqueeze to original tensor
    {
        Tensor t({2, 3});
        t.fill_inplace(0.0f);

        Tensor u = t.unsqueeze(0); // shape {1,2,3}

        // Run backward on the view alone
        std::cout << "[INFO] calling u.backward() for unsqueeze backward test\n";
        u.backward();
        std::cout << "[INFO] returned from u.backward()\n";

        std::cout << "[INFO] checking t.m_node->m_grad pointer\n";
        std::cout << "[INFO] t.m_node->m_grad is " << (t.m_node->m_grad ? "set" : "null") << "\n";

        Tensor g = t.grad();

        std::cout << "[INFO] got g from t.grad(), shape size=" << g.shape().size() << "\n";

        ASSERT_EQ(g.shape().size(), static_cast<size_t>(2), "grad rank for original after unsqueeze backward is 2");
        ASSERT_EQ(g.shape()[0], static_cast<size_t>(2), "grad dim0 == 2");
        ASSERT_EQ(g.shape()[1], static_cast<size_t>(3), "grad dim1 == 3");

        // All gradients should be ones (since out.grad is ones)
        ASSERT_EQ(g[{0,0}], 1.0f, "grad preserved at 0,0 after unsqueeze backward");
        ASSERT_EQ(g[{1,2}], 1.0f, "grad preserved at 1,2 after unsqueeze backward");
    }
}

#endif
