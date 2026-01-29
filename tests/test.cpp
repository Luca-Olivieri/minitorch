#include <iostream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <source_location>

#include "src/core/tensors.h"

int failed_tests = 0;

// A generic test function
template <typename T>
void ASSERT_EQ(T actual, T expected, std::string_view message, 
               const std::source_location location = std::source_location::current()) {
    if (actual != expected) {
        std::cerr << "[FAIL] " << message << "\n"
                  << "       Expected: " << expected << ", Got: " << actual << "\n"
                  << "       File: " << location.file_name() << "(" 
                  << location.line() << ")\n";
        failed_tests++;
    } else {
        std::cout << "[PASS] " << message << "\n";
    }
}

// int main() {
//     // Tests
//     ASSERT_EQ(1 + 1, 2, "Addition check");
//     ASSERT_EQ(10 * 10, 100, "Multiplication check");
//     ASSERT_EQ(5 - 2, 0, "Subtraction check (Intentional Fail)");

//     if (failed_tests == 0) {
//         std::cout << "\nAll tests passed!\n";
//         return 0;
//     } else {
//         std::cout << "\n" << failed_tests << " tests failed.\n";
//         return 1;
//     }
// }

// --- The Test Helper ---
// 1. Try running the code.
// 2. If it works (no throw), print FAIL.
// 3. Catch the specific exception type -> PASS.
// 4. Catch anything else -> FAIL.
#define ASSERT_THROWS(expression, exception_type) \
    do { \
        bool caught_correctly = false; \
        bool caught_other = false; \
        try { \
            expression; \
        } catch (const exception_type&) { \
            caught_correctly = true; \
        } catch (...) { \
            caught_other = true; \
        } \
        if (caught_correctly) { \
            std::cout << "[PASS] " #expression " threw " #exception_type "\n"; \
        } else { \
            std::cerr << "[FAIL] " #expression "\n" \
                      << "       Expected: " #exception_type ", Got: " << (caught_other ? "Wrong Exception" : "No Exception") << "\n" \
                      << "       File: " << __FILE__ << "(" << __LINE__ << ")\n"; \
            failed_tests++; \
        } \
    } while(0)

// --- Code Under Test ---
int divide(int a, int b) {
    if (b == 0) throw std::invalid_argument("Division by zero");
    return a / b;
}

void test_tensors_with_dims0() {
    // no tensor with 0 dims
    ASSERT_THROWS(Tensor t({0}), std::invalid_argument);
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
        Tensor t({}); // Scalar
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
    // Test 1: Should pass
    // ASSERT_THROWS(divide(10, 0), std::invalid_argument);
    
    // // Test 2: Should fail (because 10/2 is valid and won't throw)
    // ASSERT_THROWS(divide(10, 2), std::invalid_argument);
    
    // // Test 3: Should fail (wrong exception type, e.g. checking for runtime_error)
    // ASSERT_THROWS(divide(10, 0), std::runtime_error);
    
    // ASSERT_EQ(1 + 1, 2, "Addition check");
    // ASSERT_EQ(10 * 10, 100, "Multiplication check");
    // ASSERT_EQ(5 - 2, 0, "Subtraction check (Intentional Fail)");
    
    // --- [ tensor.h tests ]  
    test_tensors_with_dims0();
    test_tensor_access_errors();
    
    if (failed_tests == 0) {
        std::cout << "\nAll tests passed!\n";
        return 0;
    } else {
        std::cout << "\n" << failed_tests << " tests failed.\n";
        return 1;
    }
    
    return 0;
}
