#ifndef TEST_ADD_H
#define TEST_ADD_H

#include "src/core/tensors.h"
#include "test_utils.h"

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
}

#endif
