#ifndef TEST_NEG_H
#define TEST_NEG_H

#include "src/core/tensors.h"
#include "tests/test_utils.h"

void test_tensor_neg() {
    
    std::cout << "\n===[ test_neg.h ]===\n";

    // 1. Simple Negation
    {
        Tensor t1({3});
        t1.fill_inplace(2.0f);
        
        Tensor t2 = -t1;
        
        ASSERT_EQ(t2[{0}], -2.0f, "-(2) = -2 at index 0");
        ASSERT_EQ(t2[{1}], -2.0f, "-(2) = -2 at index 1");
    }

    // 2. Double Negation
    {
        Tensor t1({2});
        t1.fill_inplace(5.0f);
        
        Tensor t2 = -(-t1);
        ASSERT_EQ(t2[{0}], 5.0f, "-(-5) = 5");
    }

    // 3. Gradient Check (1D)
    {
        Tensor t1({3});
        t1.fill_inplace(2.0f);
        
        Tensor t2 = -t1;
        
        t2.backward();
        
        Tensor g1 = t1.grad();
        
        ASSERT_EQ(g1[{0}], -1.0f, "grad w.r.t t1 at 0 is -1");
        ASSERT_EQ(g1[{1}], -1.0f, "grad w.r.t t1 at 1 is -1");
        ASSERT_EQ(g1[{2}], -1.0f, "grad w.r.t t1 at 2 is -1");
    }
}

#endif
