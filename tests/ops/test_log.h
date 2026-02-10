#ifndef TEST_LOG_H
#define TEST_LOG_H

#include <cmath>
#include <numbers>

#include "src/core/tensors.h"
#include "tests/test_utils.h"

void test_tensor_log() {
    
    std::cout << "\n===[ test_log.h ]===\n";

    // 1. Simple Log
    {
        Tensor x({1});
        x.fill_inplace(std::exp(1.0f));
        Tensor y = x.log();
        ASSERT_EQ_APPROX(y.item(), 1.0f, 1e-4, "ln(e) = 1");
    }

    // 2. Gradient w.r.t x (ln(x)) => 1/x
    {
        Tensor x({1});
        x.fill_inplace(2.0f);
        Tensor y = x.log();
        y.backward();
        
        // expected dy/dx = 1/2 = 0.5
        ASSERT_EQ_APPROX(x.grad().item(), 0.5f, 1e-4, "d(ln x)/dx at x=2");
    }

    // 3. Chain Rule z = ln(x^2) = 2 ln(x)
    // dz/dx = 2/x. at x=3 => 2/3
    {
        Tensor x({1});
        x.fill_inplace(3.0f);
        Tensor two({1});
        two.fill_inplace(2.0f);
        
        Tensor z = x.pow(two).log();
        z.backward();
        
        ASSERT_EQ_APPROX(x.grad().item(), 2.0f/3.0f, 1e-4, "d(ln(x^2))/dx at x=3 is 2/3");
    }
}
#endif
