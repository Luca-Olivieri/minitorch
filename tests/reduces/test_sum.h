#ifndef TEST_SUM_H
#define TEST_SUM_H

#include "src/core/tensors.h"
#include "tests/test_utils.h"

void test_storage_sum() {
    std::cout << "\n===[ test_sum.h ]===\n";

    // 1. 1D reduction -> scalar
    {
        Tensor a = Tensor::linspace({4}, 1.0f, 4.0f); // [1,2,3,4]
        Tensor s = a.sum(0);
        ASSERT_EQ(s.shape().size(), (size_t)0, "1D sum produces scalar shape");
        ASSERT_EQ(s.item(), 10.0f, "sum of 1..4 == 10");
    }

    // 2. 2D reduction along dim 0
    {
        Tensor a = Tensor::linspace({2, 3}, 1.0f, 6.0f);
        // fill rows: [1,2,3], [4,5,6]

        Tensor s0 = a.sum(0); // shape {3}
        ASSERT_EQ(s0.shape().size(), (size_t)1, "2D sum dim0 yields 1D shape");
        ASSERT_EQ(s0[std::vector<size_t>{0}], 5.0f, "sum col0 == 1+4");
        ASSERT_EQ(s0[std::vector<size_t>{1}], 7.0f, "sum col1 == 2+5");
        ASSERT_EQ(s0[std::vector<size_t>{2}], 9.0f, "sum col2 == 3+6");
    }

    // 3. 2D reduction along dim 1
    {
        Tensor a = Tensor::linspace({2, 3}, 1.0f, 6.0f);

        Tensor s1 = a.sum(1); // shape {2}
        ASSERT_EQ(s1.shape().size(), (size_t)1, "2D sum dim1 yields 1D shape");
        ASSERT_EQ(s1[std::vector<size_t>{0}], 6.0f, "sum row0 == 1+2+3");
        ASSERT_EQ(s1[std::vector<size_t>{1}], 15.0f, "sum row1 == 4+5+6");
    }

    // 4. Invalid dimension
    {
        Tensor a({2, 2});
        ASSERT_THROWS(a.sum(2), std::invalid_argument);
    }

    // 5. Backward: 1D reduction -> scalar should propagate ones
    {
        Tensor a = Tensor::linspace({4}, 1.0f, 4.0f); // [1,2,3,4]
        Tensor s = a.sum(0);
        s.backward(); // sets output grad to ones

        Tensor g = a.grad();
        ASSERT_EQ(g.shape().size(), (size_t)1, "grad shape for 1D input");
        ASSERT_EQ(g.shape()[0], (size_t)4, "grad length == 4");
        for (size_t i = 0; i < 4; i++) {
            ASSERT_EQ(g[std::vector<size_t>{i}], 1.0f, "each input grad == 1");
        }
    }

    // 6. Backward: 2D reduction along dim 0 should broadcast back to full matrix
    {
        Tensor a = Tensor::linspace({2, 3}, 1.0f, 6.0f);
        Tensor s0 = a.sum(0); // shape {3}
        s0.backward();

        Tensor g = a.grad();
        ASSERT_EQ(g.shape().size(), (size_t)2, "grad shape for 2D input");
        ASSERT_EQ(g.shape()[0], (size_t)2, "rows == 2");
        ASSERT_EQ(g.shape()[1], (size_t)3, "cols == 3");
        for (size_t r = 0; r < 2; r++) {
            for (size_t c = 0; c < 3; c++) {
                ASSERT_EQ(g[std::vector<size_t>{r, c}], 1.0f, "each input grad == 1");
            }
        }
    }

    // 7. Backward: 2D reduction along dim 1 then multiplied -> gradients scale accordingly
    {
        Tensor a = Tensor::linspace({2, 3}, 1.0f, 6.0f);
        Tensor s1 = a.sum(1); // shape {2}
        Tensor two({2}, 2.0f);
        Tensor t = s1 * two; // elementwise scale of the reduced tensor
        t.backward();

        Tensor g = a.grad();
        ASSERT_EQ(g.shape().size(), (size_t)2, "grad shape for 2D input (scaled)");
        for (size_t r = 0; r < 2; r++) {
            for (size_t c = 0; c < 3; c++) {
                ASSERT_EQ(g[std::vector<size_t>{r, c}], 2.0f, "each input grad == 2 due to scaling");
            }
        }
    }
}

#endif
