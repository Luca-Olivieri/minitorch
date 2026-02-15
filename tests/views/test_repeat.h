#ifndef TEST_REPEAT_H
#define TEST_REPEAT_H

#include "src/core/tensors.h"
#include "tests/test_utils.h"

void test_repeat() {
    std::cout << "\n===[ test_repeat.h ]===\n";

    // 1. Repeat a leading singleton dimension
    {
        Tensor t = Tensor::linspace({1, 2, 3}, 0.0f, 5.0f); // shape {1,2,3}

        Tensor r = t.repeat(0, 4); // shape -> {4,2,3}

        ASSERT_EQ(r.shape().size(), static_cast<size_t>(3), "repeated rank is 3");
        ASSERT_EQ(r.shape()[0], static_cast<size_t>(4), "repeated dim 0 == 4");
        ASSERT_EQ(r.shape()[1], static_cast<size_t>(2), "repeated dim 1 == 2");
        ASSERT_EQ(r.shape()[2], static_cast<size_t>(3), "repeated dim 2 == 3");

        ASSERT_EQ(r[{0,0,0}], t[{0,0,0}], "value preserved after repeat at 0,0,0");
        ASSERT_EQ(r[{3,1,2}], t[{0,1,2}], "value preserved after repeat at 3,1,2");
    }

    // 2. Repeat on non-singleton dimension should throw
    {
        Tensor t({2, 3});
        ASSERT_THROWS(t.repeat(0, 3), std::invalid_argument);
    }

    // 3. Out-of-range repeat should throw
    {
        Tensor t({1, 2});
        ASSERT_THROWS(t.repeat(3, 2), std::invalid_argument);
    }

    // 4. Backward: gradient should accumulate (sum) over repeated dim
    {
        Tensor t = Tensor::linspace({1, 2, 3}, 0.0f, 5.0f); // shape {1,2,3}

        Tensor r = t.repeat(0, 5); // shape {5,2,3}

        // Run backward on the repeated view
        r.backward();

        Tensor g = t.grad();

        ASSERT_EQ(g.shape().size(), static_cast<size_t>(3), "grad rank for original after repeat backward is 3");
        ASSERT_EQ(g.shape()[0], static_cast<size_t>(1), "grad dim0 == 1");
        ASSERT_EQ(g.shape()[1], static_cast<size_t>(2), "grad dim1 == 2");
        ASSERT_EQ(g.shape()[2], static_cast<size_t>(3), "grad dim2 == 3");

        // Since out.grad is ones, the gradient on the original should be 'times' (5)
        ASSERT_EQ(g[{0,0,0}], 5.0f, "grad accumulated at 0,0,0 after repeat backward");
        ASSERT_EQ(g[{0,1,2}], 5.0f, "grad accumulated at 0,1,2 after repeat backward");
    }
}

#endif
