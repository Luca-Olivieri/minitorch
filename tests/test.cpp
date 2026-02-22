#include <iostream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <source_location>

#include "ops/test_add.h"
#include "ops/test_neg.h"
#include "ops/test_sub.h"
#include "ops/test_mult.h"
#include "ops/test_matmul.h"
#include "ops/test_div.h"
#include "ops/test_pow.h"
#include "ops/test_log.h"
#include "ops/test_ops.h"
#include "views/test_unsqueeze.h"
#include "views/test_squeeze.h"
#include "views/test_repeat.h"
#include "reduces/test_sum.h"
#include "reduces/test_mean.h"
#include "nn/activations/test_ReLU.h"
#include "nn/losses/test_MSELoss.h"
#include "nn/losses/test_CrossEntropyLoss.h"
#include "nn/test_nn.h"

void test_tensors_with_dims0() {
    // no tensor with 0 dims
    ASSERT_THROWS(Tensor({1,0}), std::invalid_argument);
    ASSERT_THROWS(Tensor({4, 0}), std::invalid_argument);
}

void test_tensor_access_errors() {
    // 1. Out of Bounds
    {
        Tensor t({2, 3});
        // Valid access: t[{1, 2}]
        // Invalid accesses:
        ASSERT_THROWS((t[{2, 0}]), std::out_of_range); // Dim 0 overflow
        ASSERT_THROWS((t[{0, 3}]), std::out_of_range); // Dim 1 overflow
        ASSERT_THROWS((t[{2, 3}]), std::out_of_range); // Both
    }

    // 2. Scalar Access
    {
        std::vector<size_t> empty_vec {};
        Tensor t(empty_vec); // Scalar
        // Scalars cannot be accessed by index
        ASSERT_THROWS((t[{0}]), std::invalid_argument); 
        ASSERT_THROWS((t[{0, 0}]), std::invalid_argument);
    }

    // 3. Index Size Mismatch
    {
        Tensor t({2, 2});
        ASSERT_THROWS((t[{0}]), std::invalid_argument);       // Too few indices
        ASSERT_THROWS((t[{0, 0, 0}]), std::invalid_argument); // Too many indices
    }

    // 4. item() on Non-Singleton
    {
        Tensor t({2, 2});
        ASSERT_THROWS(t.item(), std::runtime_error);
    }
}


// --- Main ---
int main() {
    test_tensors_with_dims0();
    test_tensor_access_errors();
    test_tensor_add();
    test_tensor_neg();
    test_tensor_sub();
    test_tensor_mult();
    test_tensor_div();
    test_tensor_pow();
    test_chained_ops();
    test_chained_ops_more();
    test_tensor_log();
    test_unsqueeze();
    test_squeeze();
    test_repeat();
    test_relu_forward_backward();
    test_storage_sum();
    test_storage_mean();
    test_tensor_matmul();
    test_linear_relu_forward_backward();
    test_two_layer_linear_relu_linear_backward();
    test_mse_loss_forward_backward();
    test_crossentropy_loss_forward_backward();
    
    if (failed_tests == 0) {
        std::cout << "\nAll tests passed!\n";
        return 0;
    } else {
        std::cout << "\n" << failed_tests << " tests failed.\n";
        return 1;
    }
    
    return 0;
}
