#ifndef TEST_SUB_H
#define TEST_SUB_H

#include "src/core/tensors.h"
#include "tests/test_utils.h"

void test_tensor_sub() {
    
    std::cout << "\n===[ test_sub.h ]===\n";

    // 1. Simple 1D subtraction
    {
        Tensor t1({3});
        t1.fill_inplace(5.0f);
        Tensor t2({3});
        t2.fill_inplace(2.0f);
        
        Tensor t3 = t1 - t2;
        
        ASSERT_EQ(t3[{0}], 3.0f, "5-2=3 at index 0");
        ASSERT_EQ(t3[{1}], 3.0f, "5-2=3 at index 1");
        ASSERT_EQ(t3[{2}], 3.0f, "5-2=3 at index 2");
    }

    // 2. Simple 2D subtraction
    {
        Tensor t1({2, 2});
        t1.fill_inplace(50.0f);
        Tensor t2({2, 2});
        t2.fill_inplace(20.0f);

        Tensor t3 = t1 - t2;
        
        ASSERT_EQ(t3[{0, 0}], 30.0f, "50-20=30 at index 0,0");
        ASSERT_EQ(t3[{1, 1}], 30.0f, "50-20=30 at index 1,1");
    }

    // 3. Shape Mismatch
    {
        Tensor t1({2, 2});
        Tensor t2({2, 3});
        ASSERT_THROWS(t1 - t2, std::invalid_argument);
    }

    // 4. Gradient Check (1D) Use Chain Rule
    {
        Tensor t1({3});
        t1.fill_inplace(5.0f);
        Tensor t2({3});
        t2.fill_inplace(2.0f);
        
        Tensor t3 = t1 - t2;
        
        t3.backward();
        
        Tensor g1 = t1.grad();
        Tensor g2 = t2.grad();
        
        ASSERT_EQ(g1[{0}], 1.0f, "grad w.r.t t1 at 0 is 1");
        ASSERT_EQ(g1[{1}], 1.0f, "grad w.r.t t1 at 1 is 1");
        ASSERT_EQ(g1[{2}], 1.0f, "grad w.r.t t1 at 2 is 1");

        ASSERT_EQ(g2[{0}], -1.0f, "grad w.r.t t2 at 0 is -1");
        ASSERT_EQ(g2[{1}], -1.0f, "grad w.r.t t2 at 1 is -1");
        ASSERT_EQ(g2[{2}], -1.0f, "grad w.r.t t2 at 2 is -1");
    }

    // 5. Chain Rule (2 layers)
    {
        Tensor a({1}); a.fill_inplace(10.0f);
        Tensor b({1}); b.fill_inplace(3.0f);
        
        Tensor c = a - b; // 7
        Tensor d = c - a; // 7 - 10 = -3
        
        // d = (a - b) - a = -b
        // d_d/d_a = 0
        // d_d/d_b = -1
        
        d.backward();
        
        Tensor grad_a = a.grad();
        Tensor grad_b = b.grad();
        
        ASSERT_EQ(grad_a.item(), 0.0f, "grad a (d = -b) is 0");
        ASSERT_EQ(grad_b.item(), -1.0f, "grad b (d = -b) is -1");
    }
}

#endif
