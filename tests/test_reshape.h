#ifndef TEST_RESHAPE_H
#define TEST_RESHAPE_H

#include <iostream>

#include "test_utils.h"
#include "src/core/tensors.h"

void test_reshape_gradients() {
    std::cout << "\nRunning reshape gradient tests...\n";
    
    // Test 1: Simple reshape - 1D to 2D
    {
        Tensor x({6});
        x.linspace(1.0f, 6.0f);
        Tensor y = x.reshape({2, 3});
        y.backward();
        
        // Gradient should backprop through reshape to original shape
        ASSERT_EQ(x.grad().shape()[0], (size_t)6, "Gradient shape after reshape 1D->2D");
        for (size_t i = 0; i < 6; i++) {
            ASSERT_EQ(x.grad()[{i}], 1.0f, "Gradient value at index " + std::to_string(i));
        }
    }
    
    // Test 2: 2D to 1D reshape
    {
        Tensor x({2, 3});
        x.linspace(1.0f, 6.0f);
        Tensor y = x.reshape({6});
        y.backward();
        
        ASSERT_EQ(x.grad().shape()[0], (size_t)2, "Gradient shape dim 0 after reshape 2D->1D");
        ASSERT_EQ(x.grad().shape()[1], (size_t)3, "Gradient shape dim 1 after reshape 2D->1D");
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 3; j++) {
                ASSERT_EQ(x.grad()[{i, j}], 1.0f, 
                    "Gradient value at [" + std::to_string(i) + "," + std::to_string(j) + "]");
            }
        }
    }
    
    // Test 3: Reshape with arithmetic operation
    {
        Tensor x({4});
        x.linspace(1.0f, 4.0f);
        Tensor y = x.reshape({2, 2});
        
        Tensor z({2, 2});
        z.fill(2.0f);
        
        Tensor out = y * z;
        out.backward();
        
        // Gradient through multiplication and reshape
        // d(out)/d(x) = z, reshaped back to original shape
        ASSERT_EQ(x.grad().shape()[0], (size_t)4, "Gradient shape after reshape + multiply");
        for (size_t i = 0; i < 4; i++) {
            ASSERT_EQ(x.grad()[{i}], 2.0f, "Gradient value after multiply at index " + std::to_string(i));
        }
    }
    
    // Test 4: Multiple reshapes in chain
    {
        Tensor x({12});
        x.linspace(1.0f, 12.0f);
        Tensor y = x.reshape({3, 4});
        Tensor z = y.reshape({2, 6});
        z.backward();
        
        ASSERT_EQ(x.grad().shape()[0], (size_t)12, "Gradient shape after chained reshapes");
        for (size_t i = 0; i < 12; i++) {
            ASSERT_EQ(x.grad()[{i}], 1.0f, "Gradient value after chained reshapes at index " + std::to_string(i));
        }
    }
    
    // Test 5: Reshape with addition
    {
        Tensor x({6});
        x.fill(1.0f);
        
        Tensor y = x.reshape({2, 3});
        
        Tensor b({2, 3});
        b.fill(3.0f);
        
        Tensor out = y + b;
        out.backward();
        
        // Gradient of addition is 1.0, should backprop through reshape
        ASSERT_EQ(x.grad().shape()[0], (size_t)6, "Gradient shape after reshape + add");
        for (size_t i = 0; i < 6; i++) {
            ASSERT_EQ(x.grad()[{i}], 1.0f, "Gradient after addition at index " + std::to_string(i));
        }
    }
    
    // Test 6: Complex reshape dimensions
    {
        Tensor x({24});
        x.linspace(1.0f, 24.0f);
        Tensor y = x.reshape({2, 3, 4});
        y.backward();
        
        ASSERT_EQ(x.grad().shape()[0], (size_t)24, "Gradient shape after 1D->3D reshape");
        for (size_t i = 0; i < 24; i++) {
            ASSERT_EQ(x.grad()[{i}], 1.0f, "Gradient value after 3D reshape at index " + std::to_string(i));
        }
    }
    
    // Test 7: Reshape with power operation
    {
        Tensor x({4});
        x.fill(2.0f);
        
        Tensor y = x.reshape({2, 2});
        
        Tensor exp({2, 2});
        exp.fill(3.0f);
        
        Tensor out = y.pow(exp);
        out.backward();
        
        // d(x^3)/dx = 3*x^2 = 3*2^2 = 12
        ASSERT_EQ(x.grad().shape()[0], (size_t)4, "Gradient shape after reshape + pow");
        for (size_t i = 0; i < 4; i++) {
            ASSERT_EQ(x.grad()[{i}], 12.0f, "Gradient after power at index " + std::to_string(i));
        }
    }
    
    // Test 8: Scalar to multidimensional
    {
        std::vector<size_t> scalar_shape {};
        Tensor x(scalar_shape);
        x.fill(5.0f);
        
        Tensor y = x.reshape({1, 1});
        y.backward();
        
        ASSERT_EQ(x.grad().shape().size(), (size_t)0, "Gradient is scalar after scalar->2D reshape");
        ASSERT_EQ(x.grad().item(), 1.0f, "Scalar gradient value");
    }
}

#endif
