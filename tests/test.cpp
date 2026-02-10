#include <iostream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <source_location>

#include "test_higher_order.h"
#include "test_add.h"
#include "test_neg.h"
#include "test_sub.h"
#include "test_mult.h"

void test_tensors_with_dims0() {
    // no tensor with 0 dims
    ASSERT_THROWS(Tensor t({0, 1}), std::invalid_argument);
    ASSERT_THROWS(Tensor t({4, 0}), std::invalid_argument);
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
    test_higher_order_derivatives();
    
    if (failed_tests == 0) {
        std::cout << "\nAll tests passed!\n";
        return 0;
    } else {
        std::cout << "\n" << failed_tests << " tests failed.\n";
        return 1;
    }
    
    return 0;
}
