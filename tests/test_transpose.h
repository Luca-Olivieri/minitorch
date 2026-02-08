#ifndef TEST_TRANSPOSE_H
#define TEST_TRANSPOSE_H

#include <iostream>

#include "test_utils.h"
#include "src/core/tensors.h"

void test_transpose_gradients() {
    std::cout << "\nRunning transpose gradient tests...\n";
    
    // Test 1: Simple 2D transpose
    {
        Tensor x({2, 3});
        x.linspace_inplace(1.0f, 6.0f);
        Tensor y = x.transpose(0, 1);
        
        ASSERT_EQ(y.shape()[0], (size_t)3, "Transposed tensor dim 0");
        ASSERT_EQ(y.shape()[1], (size_t)2, "Transposed tensor dim 1");
        
        y.backward();
        
        ASSERT_EQ(x.grad().shape()[0], (size_t)2, "Gradient shape dim 0 after transpose");
        ASSERT_EQ(x.grad().shape()[1], (size_t)3, "Gradient shape dim 1 after transpose");
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                ASSERT_EQ(x.grad()[{i, j}], 1.0f, 
                    "Gradient value at [" + std::to_string(i) + "," + std::to_string(j) + "]");
            }
        }
    }
    
    // Test 2: 3D transpose - swap dim 0 and 2
    {
        Tensor x({2, 3, 4});
        x.linspace_inplace(1.0f, 24.0f);
        Tensor y = x.transpose(0, 2);
        
        ASSERT_EQ(y.shape()[0], (size_t)4, "3D transpose dim 0");
        ASSERT_EQ(y.shape()[1], (size_t)3, "3D transpose dim 1");
        ASSERT_EQ(y.shape()[2], (size_t)2, "3D transpose dim 2");
        
        y.backward();
        
        ASSERT_EQ(x.grad().shape()[0], (size_t)2, "3D gradient shape dim 0");
        ASSERT_EQ(x.grad().shape()[1], (size_t)3, "3D gradient shape dim 1");
        ASSERT_EQ(x.grad().shape()[2], (size_t)4, "3D gradient shape dim 2");
    }
    
    // Test 3: Transpose with addition
    {
        Tensor x({3, 2});
        x.fill_inplace(2.0f);
        
        Tensor y = x.transpose(0, 1);
        
        Tensor b({2, 3});
        b.fill_inplace(3.0f);
        
        Tensor out = y + b;
        out.backward();
        
        ASSERT_EQ(x.grad().shape()[0], (size_t)3, "Gradient shape after transpose + add dim 0");
        ASSERT_EQ(x.grad().shape()[1], (size_t)2, "Gradient shape after transpose + add dim 1");
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 2; j++) {
                ASSERT_EQ(x.grad()[{i, j}], 1.0f, 
                    "Gradient after addition at [" + std::to_string(i) + "," + std::to_string(j) + "]");
            }
        }
    }
    
    // Test 4: Transpose with multiplication
    {
        Tensor x({2, 3});
        x.linspace_inplace(1.0f, 6.0f);
        
        Tensor y = x.transpose(0, 1);
        
        Tensor b({3, 2});
        b.fill_inplace(2.0f);
        
        Tensor out = y * b;
        out.backward();
        
        ASSERT_EQ(x.grad().shape()[0], (size_t)2, "Gradient shape after transpose + multiply dim 0");
        ASSERT_EQ(x.grad().shape()[1], (size_t)3, "Gradient shape after transpose + multiply dim 1");
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                ASSERT_EQ(x.grad()[{i, j}], 2.0f, 
                    "Gradient after multiply at [" + std::to_string(i) + "," + std::to_string(j) + "]");
            }
        }
    }
    
    // Test 5: Chained transposes (should cancel out)
    {
        Tensor x({2, 3});
        x.fill_inplace(5.0f);
        
        Tensor y = x.transpose(0, 1);
        Tensor z = y.transpose(0, 1);
        
        ASSERT_EQ(z.shape()[0], (size_t)2, "Chained transpose dim 0 (should be original)");
        ASSERT_EQ(z.shape()[1], (size_t)3, "Chained transpose dim 1 (should be original)");
        
        z.backward();
        
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                ASSERT_EQ(x.grad()[{i, j}], 1.0f, 
                    "Gradient after chained transpose at [" + std::to_string(i) + "," + std::to_string(j) + "]");
            }
        }
    }
    
    // Test 6: Transpose with power operation
    {
        Tensor x({2, 2});
        x.fill_inplace(2.0f);
        
        Tensor y = x.transpose(0, 1);
        
        Tensor exp({2, 2});
        exp.fill_inplace(3.0f);
        
        Tensor out = y.pow(exp);
        out.backward();
        
        // d(x^3)/dx = 3*x^2 = 3*2^2 = 12
        ASSERT_EQ(x.grad().shape()[0], (size_t)2, "Gradient shape after transpose + pow dim 0");
        ASSERT_EQ(x.grad().shape()[1], (size_t)2, "Gradient shape after transpose + pow dim 1");
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 2; j++) {
                ASSERT_EQ(x.grad()[{i, j}], 12.0f, 
                    "Gradient after power at [" + std::to_string(i) + "," + std::to_string(j) + "]");
            }
        }
    }
    
    // Test 7: Transpose middle dimensions (3D)
    {
        Tensor x({2, 3, 4});
        x.linspace_inplace(1.0f, 24.0f);
        
        Tensor y = x.transpose(1, 2);
        
        ASSERT_EQ(y.shape()[0], (size_t)2, "3D transpose middle dims - dim 0");
        ASSERT_EQ(y.shape()[1], (size_t)4, "3D transpose middle dims - dim 1");
        ASSERT_EQ(y.shape()[2], (size_t)3, "3D transpose middle dims - dim 2");
        
        y.backward();
        
        ASSERT_EQ(x.grad().shape()[0], (size_t)2, "3D gradient shape after middle transpose dim 0");
        ASSERT_EQ(x.grad().shape()[1], (size_t)3, "3D gradient shape after middle transpose dim 1");
        ASSERT_EQ(x.grad().shape()[2], (size_t)4, "3D gradient shape after middle transpose dim 2");
    }
    
    // Test 8: Transpose with subtraction
    {
        Tensor x({2, 3});
        x.linspace_inplace(1.0f, 6.0f);
        
        Tensor y = x.transpose(0, 1);
        
        Tensor b({3, 2});
        b.fill_inplace(2.0f);
        
        Tensor out = y - b;
        out.backward();
        
        ASSERT_EQ(x.grad().shape()[0], (size_t)2, "Gradient shape after transpose - subtract dim 0");
        ASSERT_EQ(x.grad().shape()[1], (size_t)3, "Gradient shape after transpose - subtract dim 1");
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                ASSERT_EQ(x.grad()[{i, j}], 1.0f, 
                    "Gradient after subtract at [" + std::to_string(i) + "," + std::to_string(j) + "]");
            }
        }
    }
    
    // Test 10: Transpose followed by reshape
    {
        Tensor x({2, 3});
        x.linspace_inplace(1.0f, 6.0f);
        
        Tensor y = x.transpose(0, 1);  // Shape becomes {3, 2}
        Tensor z = y.reshape({6});      // Shape becomes {6}
        
        ASSERT_EQ(z.shape()[0], (size_t)6, "Shape after transpose + reshape");
        
        z.backward();
        
        ASSERT_EQ(x.grad().shape()[0], (size_t)2, "Gradient shape after transpose + reshape dim 0");
        ASSERT_EQ(x.grad().shape()[1], (size_t)3, "Gradient shape after transpose + reshape dim 1");
    }
}

#endif
