#ifndef TEST_ADD_H
#define TEST_ADD_H

#include "src/core/tensors.h"
#include "tests/test_utils.h"

void test_tensor_add() {
    
    std::cout << "\n===[ test_add.h ]===\n";

    // 1. Simple 1D addition
    {
        Tensor t1({3});
        t1.fill_inplace(1.0f);
        Tensor t2({3});
        t2.fill_inplace(2.0f);
        
        Tensor t3 = t1 + t2;
        
        ASSERT_EQ(t3[{0}], 3.0f, "1+2=3 at index 0");
        ASSERT_EQ(t3[{1}], 3.0f, "1+2=3 at index 1");
        ASSERT_EQ(t3[{2}], 3.0f, "1+2=3 at index 2");
    }

    // 2. Simple 2D addition
    {
        Tensor t1({2, 2});
        t1.fill_inplace(10.0f);
        Tensor t2({2, 2});
        t2.fill_inplace(20.0f);

        Tensor t3 = t1 + t2;
        
        ASSERT_EQ(t3[{0, 0}], 30.0f, "10+20=30 at index 0,0");
        ASSERT_EQ(t3[{1, 1}], 30.0f, "10+20=30 at index 1,1");
    }

    // 3. Shape Mismatch
    {
        Tensor t1({2, 2});
        Tensor t2({2, 3});
        ASSERT_THROWS(t1 + t2, std::invalid_argument);
    }

    // 4. Gradient Check (1D)
    {
        Tensor t1({3});
        t1.fill_inplace(1.0f);
        Tensor t2({3});
        t2.fill_inplace(2.0f);
        
        Tensor t3 = t1 + t2;
        
        t3.backward();
        
        Tensor g1 = t1.grad();
        Tensor g2 = t2.grad();
        
        ASSERT_EQ(g1[{0}], 1.0f, "grad w.r.t t1 at 0 is 1");
        ASSERT_EQ(g1[{1}], 1.0f, "grad w.r.t t1 at 1 is 1");
        ASSERT_EQ(g1[{2}], 1.0f, "grad w.r.t t1 at 2 is 1");

        ASSERT_EQ(g2[{0}], 1.0f, "grad w.r.t t2 at 0 is 1");
        ASSERT_EQ(g2[{1}], 1.0f, "grad w.r.t t2 at 1 is 1");
        ASSERT_EQ(g2[{2}], 1.0f, "grad w.r.t t2 at 2 is 1");
    }

    // 5. Gradient Check (Self Addition)
    {
        Tensor t1({3});
        t1.fill_inplace(10.0f);
        
        Tensor t2 = t1 + t1;
        t2.backward();
        
        Tensor g1 = t1.grad();
        
        ASSERT_EQ(g1[{0}], 2.0f, "grad w.r.t t1 (self add) at 0 is 2");
    }

    // 6. Chain Rule (2 layers)
    {
        Tensor a({1}); a.fill_inplace(2.0f);
        Tensor b({1}); b.fill_inplace(3.0f);
        
        Tensor c = a + b; // 5
        Tensor d = c + a; // 5 + 2 = 7
        
        d.backward();
        
        Tensor grad_a = a.grad();
        Tensor grad_b = b.grad();
        
        ASSERT_EQ(grad_a.item(), 2.0f, "grad a (d = 2a+b) is 2");
        ASSERT_EQ(grad_b.item(), 1.0f, "grad b (d = 2a+b) is 1");
    }
}

#endif
