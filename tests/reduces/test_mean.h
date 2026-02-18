#ifndef TEST_MEAN_H
#define TEST_MEAN_H

#include "src/core/tensors.h"
#include "tests/test_utils.h"

void test_storage_mean() {
    std::cout << "\n===[ test_mean.h ]===\n";

    // 1. 1D reduction -> scalar
    {
        Tensor a = Tensor::linspace({4}, 1.0f, 4.0f); // [1,2,3,4]
        Tensor m = a.mean(0);
        ASSERT_EQ(m.shape().size(), (size_t)0, "1D mean produces scalar shape");
        ASSERT_EQ_APPROX(m.item(), 2.5f, 1e-6, "mean of 1..4 == 2.5");
    }

    // 2. 2D reduction along dim 0
    {
        Tensor a = Tensor::linspace({2, 3}, 1.0f, 6.0f);
        Tensor m0 = a.mean(0); // shape {3}
        ASSERT_EQ(m0.shape().size(), (size_t)1, "2D mean dim0 yields 1D shape");
        ASSERT_EQ_APPROX(m0[std::vector<size_t>{0}], 2.5f, 1e-6, "mean col0 == 2.5");
        ASSERT_EQ_APPROX(m0[std::vector<size_t>{1}], 3.5f, 1e-6, "mean col1 == 3.5");
        ASSERT_EQ_APPROX(m0[std::vector<size_t>{2}], 4.5f, 1e-6, "mean col2 == 4.5");
    }

    // 3. 2D reduction along dim 1
    {
        Tensor a = Tensor::linspace({2, 3}, 1.0f, 6.0f);
        Tensor m1 = a.mean(1); // shape {2}
        ASSERT_EQ(m1.shape().size(), (size_t)1, "2D mean dim1 yields 1D shape");
        ASSERT_EQ_APPROX(m1[std::vector<size_t>{0}], 2.0f, 1e-6, "mean row0 == 2.0");
        ASSERT_EQ_APPROX(m1[std::vector<size_t>{1}], 5.0f, 1e-6, "mean row1 == 5.0");
    }

    // 4. Backward: 1D reduction -> scalar should propagate 1/N
    {
        Tensor a = Tensor::linspace({4}, 1.0f, 4.0f); // [1,2,3,4]
        Tensor m = a.mean(0);
        m.backward();

        Tensor g = a.grad();
        for (size_t i = 0; i < 4; i++) {
            ASSERT_EQ_APPROX(g[std::vector<size_t>{i}], 0.25f, 1e-6, "each input grad == 1/4");
        }
    }

    // 5. Backward: 2D reduction along dim 0 should broadcast back with factor 1/rows
    {
        Tensor a = Tensor::linspace({2, 3}, 1.0f, 6.0f);
        Tensor m0 = a.mean(0); // shape {3}
        m0.backward();

        Tensor g = a.grad();
        for (size_t r = 0; r < 2; r++) {
            for (size_t c = 0; c < 3; c++) {
                ASSERT_EQ_APPROX(g[std::vector<size_t>{r, c}], 0.5f, 1e-6, "each input grad == 1/2");
            }
        }
    }

    // 6. Backward: scaled reduced mean should scale gradients
    {
        Tensor a = Tensor::linspace({2, 3}, 1.0f, 6.0f);
        Tensor m1 = a.mean(1); // shape {2}
        Tensor two({2}, 2.0f);
        Tensor t = m1 * two; // elementwise scale
        t.backward();

        Tensor g = a.grad();
        for (size_t r = 0; r < 2; r++) {
            for (size_t c = 0; c < 3; c++) {
                ASSERT_EQ_APPROX(g[std::vector<size_t>{r, c}], 2.0f/3.0f, 1e-6, "each input grad == 2/3 due to scaling");
            }
        }
    }
}

#endif
