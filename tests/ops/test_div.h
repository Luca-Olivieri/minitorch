#ifndef TEST_DIV_H
#define TEST_DIV_H

#include "src/core/tensors.h"
#include "tests/test_utils.h"

void test_tensor_div() {
    
    std::cout << "\n===[ test_div.h ]===\n";

    // 1. Simple 1D division
    {
        Tensor t1({3});
        t1.fill_inplace(6.0f);
        Tensor t2({3});
        t2.fill_inplace(2.0f);
        
        Tensor t3 = t1 / t2;
        
        ASSERT_EQ(t3[{0}], 3.0f, "6/2=3 at index 0");
        ASSERT_EQ(t3[{1}], 3.0f, "6/2=3 at index 1");
        ASSERT_EQ(t3[{2}], 3.0f, "6/2=3 at index 2");
    }

    // 2. Simple 2D division
    {
        Tensor t1({2, 2});
        t1.fill_inplace(10.0f);
        Tensor t2({2, 2});
        t2.fill_inplace(2.0f);

        Tensor t3 = t1 / t2;
        
        ASSERT_EQ(t3[{0, 0}], 5.0f, "10/2=5 at index 0,0");
        ASSERT_EQ(t3[{1, 1}], 5.0f, "10/2=5 at index 1,1");
    }

    // 3. Shape Mismatch
    {
        Tensor t1({2, 2});
        Tensor t2({2, 3});
        ASSERT_THROWS(t1 / t2, std::invalid_argument);
    }

    // 4. Gradient Check (1D) z = x / y
    // dz/dx = 1/y
    // dz/dy = -x/y^2
    {
        Tensor x({3});
        x.fill_inplace(6.0f);
        Tensor y({3});
        y.fill_inplace(2.0f);
        
        Tensor z = x / y; // 3
        
        z.backward();
        
        Tensor gx = x.grad();
        Tensor gy = y.grad();
        
        // dz/dx = 1/2 = 0.5
        ASSERT_EQ(gx[{0}], 0.5f, "grad w.r.t x (x/y) at 0 is 0.5");
        ASSERT_EQ(gx[{1}], 0.5f, "grad w.r.t x (x/y) at 1 is 0.5");
        
        // dz/dy = -6/(2^2) = -6/4 = -1.5
        ASSERT_EQ(gy[{0}], -1.5f, "grad w.r.t y (x/y) at 0 is -1.5");
        ASSERT_EQ(gy[{1}], -1.5f, "grad w.r.t y (x/y) at 1 is -1.5");
    }

    // 5. Self Division z = x / x = 1
    // dz/dx = 0 (mathematically)
    // Formula check: dx/dx = 1/x + x * (-1/x^2) = 1/x - 1/x = 0
    {
        Tensor x({1});
        x.fill_inplace(2.0f);
        
        Tensor z = x / x;
        z.backward();
        
        Tensor g = x.grad();
        
        ASSERT_EQ(g[{0}], 0.0f, "grad w.r.t x (x/x) is 0");
    }
}

#endif
