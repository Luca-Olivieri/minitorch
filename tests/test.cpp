#include <iostream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <source_location>

#include "test_reshape.h"
#include "test_transpose.h"

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

// void test_dice() {
//     Tensor t({5});
//     t.fill_inplace(1.0f);
//     // Dice 0-2 (size 2)
//     t.m_node->m_storage.dice(0, 0, 2);
//     ASSERT_EQ(t.m_storage.m_shape[0], (size_t)2, "Dice shape check");
    
//     // Test invalid dice
//     TensorNode t2({5});
//     ASSERT_THROWS(t2.m_storage.dice(0, 0, 6), std::out_of_range); // OOB
//     ASSERT_THROWS(t2.m_storage.dice(0, 3, 2), std::out_of_range); // start > end
// }

// void test_is_contiguous() {
//     std::cout << "\nRunning is_contiguous tests...\n";
    
//     // Default contiguous
//     Tensor t1({2, 3});
//     ASSERT_EQ(t1.is_contiguous(), true, "New tensor should be contiguous");

//     // Transpose (1, 4) -> (4, 1) (Contiguous despite stride swap because dim is 1)
//     Tensor t2({1, 4}); 
//     t2.fill_inplace(1.0);
//     t2.m_storage.transpose(0, 1);
//     ASSERT_EQ(t2.is_contiguous(), true, "Transposed (1,4) -> (4,1) should be contiguous");
    
//     // Non-contiguous slice
//     Tensor t3({4, 4});
//     // Slice columns (dim 1)
//     t3.Tensor.slice(1, 0); // shape (4), stride (4)
//     ASSERT_EQ(t3.is_contiguous(), false, "Column slice of (4,4) should be non-contiguous");
    
//     // Contiguous slice
//     TensorNode t4({4, 4});
//     // Slice rows (dim 0)
//     t4.m_storage.slice(0, 0); // shape (4), stride (1)
//     ASSERT_EQ(t4.is_contiguous(), true, "Row slice of (4,4) should be contiguous");
// }



// --- Main ---
int main() {
    test_tensors_with_dims0();
    test_tensor_access_errors();
    test_reshape_gradients();
    test_transpose_gradients();
    // test_dice();
    // test_is_contiguous();
    
    if (failed_tests == 0) {
        std::cout << "\nAll tests passed!\n";
        return 0;
    } else {
        std::cout << "\n" << failed_tests << " tests failed.\n";
        return 1;
    }
    
    return 0;
}
