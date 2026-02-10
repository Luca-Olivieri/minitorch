#ifndef TEST_MULT_H
#define TEST_MULT_H

#include "src/core/tensors.h"
#include "test_utils.h"

void test_tensor_mult() {
    
    std::cout << "\n===[ test_mult.h ]===\n";

    // 1. Simple 1D multiplication
    {
        Tensor t1({3});
        t1.fill_inplace(2.0f);
        Tensor t2({3});
        t2.fill_inplace(3.0f);
        
        Tensor t3 = t1 * t2;
        
        ASSERT_EQ(t3[{0}], 6.0f, "2*3=6 at index 0");
        ASSERT_EQ(t3[{1}], 6.0f, "2*3=6 at index 1");
        ASSERT_EQ(t3[{2}], 6.0f, "2*3=6 at index 2");
    }

    // 2. Simple 2D multiplication
    {
        Tensor t1({2, 2});
        t1.fill_inplace(5.0f);
        Tensor t2({2, 2});
        t2.fill_inplace(5.0f);

        Tensor t3 = t1 * t2;
        
        ASSERT_EQ(t3[{0, 0}], 25.0f, "5*5=25 at index 0,0");
        ASSERT_EQ(t3[{1, 1}], 25.0f, "5*5=25 at index 1,1");
    }

    // 3. Shape Mismatch
    {
        Tensor t1({2, 2});
        Tensor t2({2, 3});
        ASSERT_THROWS(t1 * t2, std::invalid_argument);
    }

    // 4. Gradient Check (1D) z = x * y
    // dz/dx = y
    // dz/dy = x
    {
        Tensor t1({3});
        t1.fill_inplace(2.0f);
        Tensor t2({3});
        t2.fill_inplace(3.0f);
        
        Tensor t3 = t1 * t2;
        
        t3.backward();
        
        Tensor g1 = t1.grad();
        Tensor g2 = t2.grad();
        
        ASSERT_EQ(g1[{0}], 3.0f, "grad w.r.t t1 at 0 is 3");
        ASSERT_EQ(g1[{1}], 3.0f, "grad w.r.t t1 at 1 is 3");
        
        ASSERT_EQ(g2[{0}], 2.0f, "grad w.r.t t2 at 0 is 2");
        ASSERT_EQ(g2[{1}], 2.0f, "grad w.r.t t2 at 1 is 2");
    }

    // 5. Square (Self Multiplication) z = x * x
    // dz/dx = 2x
    {
        Tensor x({3});
        x.fill_inplace(4.0f);
        
        Tensor z = x * x;
        z.backward();
        
        Tensor g = x.grad();
        
        ASSERT_EQ(g[{0}], 8.0f, "grad w.r.t x (x*x) at 0 is 8 (2*4)");
    }
}

#endif
