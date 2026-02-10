#ifndef TEST_POW_H
#define TEST_POW_H

#include <cmath>

#include "src/core/tensors.h"
#include "tests/test_utils.h"

void test_tensor_pow() {
    
    std::cout << "\n===[ test_pow.h ]===\n";

    // 1. Simple 1D Power (x^2)
    {
        Tensor t1({3});
        t1.fill_inplace(3.0f);
        Tensor t2({3});
        t2.fill_inplace(2.0f);
        
        Tensor t3 = t1.pow(t2);
        
        ASSERT_EQ(t3[{0}], 9.0f, "3^2=9 at index 0");
        ASSERT_EQ(t3[{1}], 9.0f, "3^2=9 at index 1");
        ASSERT_EQ(t3[{2}], 9.0f, "3^2=9 at index 2");
    }

    // 2. Gradient Check w.r.t Base (x^n)
    // z = x^y. if y is effectively constant. dz/dx = y * x^(y-1)
    {
        Tensor x({1});
        x.fill_inplace(3.0f);
        Tensor y({1});
        y.fill_inplace(2.0f);
        
        Tensor z = x.pow(y);
        z.backward();
        
        Tensor gx = x.grad();
        // dz/dx = 2 * 3^(1) = 6
        ASSERT_EQ(gx[{0}], 6.0f, "grad w.r.t base (x^2 at x=3) is 6");
    }

    // 3. Gradient Check w.r.t Exponent (a^x)
    // z = x^y. dz/dy = x^y * ln(x)
    {
        Tensor base({1});
        base.fill_inplace(2.0f);
        Tensor exp({1});
        exp.fill_inplace(3.0f);
        
        Tensor out = base.pow(exp); // 2^3 = 8
        out.backward();
        
        Tensor g_exp = exp.grad();
        
        // expected = 2^3 * ln(2) = 8 * 0.693147... = 5.54517...
        float expected = std::pow(2.0f, 3.0f) * std::log(2.0f);
        ASSERT_EQ_APPROX(g_exp.item(), expected, 1e-4, "grad w.r.t exp (2^x at x=3)");
    }

    // 4. Gradient Check Both
    {
        Tensor x({1}); x.fill_inplace(2.0f);
        Tensor y({1}); y.fill_inplace(2.0f);
        
        Tensor z = x.pow(y); // 2^2 = 4
        z.backward();
        
        // dx: y * x^(y-1) = 2 * 2^1 = 4
        ASSERT_EQ(x.grad().item(), 4.0f, "grad base (2^2)");
        
        // dy: x^y * ln(x) = 4 * ln(2)
        float expected_dy = 4.0f * std::log(2.0f);
        ASSERT_EQ_APPROX(y.grad().item(), expected_dy, 1e-4, "grad exp (2^2)");
    }
}

#endif
