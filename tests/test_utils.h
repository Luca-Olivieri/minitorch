#include <iostream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <source_location>

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

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

#endif
