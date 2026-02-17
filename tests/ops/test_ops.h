#ifndef TEST_OPS_H
#define TEST_OPS_H

#include <cmath>

#include "src/core/tensors.h"
#include "tests/test_utils.h"

void test_chained_ops() {
    
    std::cout << "\n===[ test_ops.h ]===\n";

    // Scenario:
    // z = - log( y + (x-y)^2 / x )
    // This uses: neg, log, add, div, pow, sub, mult (implicitly in pow maybe? no explicit mult in formula? Ah x*y was in my first draft but I simplified)
    // Let's verify if I used Multi?
    // My implemented formula in previous thought: 
    // a = x * y
    // d = a + ...
    // Yes, let's use the first formula I derived to include MULT.
    
    // Formula:
    // z = - log( (x * y + (x - y)^2) / x )
    
    // x = [2.0, 4.0]
    // y = [3.0, 1.0]
    
    Tensor x({2});
    x[{0}] = 2.0f;
    x[{1}] = 4.0f;
    
    Tensor y({2});
    y[{0}] = 3.0f;
    y[{1}] = 1.0f;
    
    Tensor two({2});
    two.fill_inplace(2.0f);
    
    // Operations
    Tensor a = x * y;           // Mult
    Tensor b = x - y;           // Sub
    Tensor c = b.pow(two);      // Pow
    Tensor d = a + c;           // Add
    Tensor e = d / x;           // Div
    Tensor f = e.log();         // Log
    Tensor z = -f;              // Neg
    
    // Manual Calculation
    // Index 0 (x=2, y=3)
    // a = 6
    // b = -1
    // c = 1
    // d = 7
    // e = 3.5
    // f = log(3.5) approx 1.25276
    // z = -1.25276
    
    ASSERT_EQ_APPROX(z[{0}], -1.25276f, 1e-4, "Forward check index 0");
    
    // Index 1 (x=4, y=1)
    // a = 4
    // b = 3
    // c = 9
    // d = 13
    // e = 13/4 = 3.25
    // f = log(3.25) approx 1.17865
    // z = -1.17865
    
    ASSERT_EQ_APPROX(z[{1}], -1.17865f, 1e-4, "Forward check index 1");

    // Backward (explicitly do not retain graph)
    z.backward(false);
    
    Tensor gx = x.grad();
    Tensor gy = y.grad();
    
    // Derivative Derivation
    // z = - log( (xy + (x-y)^2)/x )
    //   = - log( y + (x-y)^2/x )
    // Let inner = y + (x-y)^2/x
    
    // dz/dx = - (1/inner) * d(inner)/dx
    // d(inner)/dx = 1 - y^2/x^2 (derived previously: (x^2 - y^2)/x^2)
    
    // Index 0 (x=2, y=3):
    // inner = 3.5
    // d(inner)/dx = 1 - 9/4 = -1.25
    // dz/dx = - (1/3.5) * (-1.25) = 1.25/3.5 = 5/14 approx 0.35714
    
    ASSERT_EQ_APPROX(gx[{0}], 0.35714f, 1e-4, "Backward grad x index 0");
    
    // dz/dy = - (1/inner) * d(inner)/dy
    // d(inner)/dy = -1 + 2y/x (derived previously)
    
    // Index 0 (x=2, y=3):
    // d(inner)/dy = -1 + 6/2 = 2
    // dz/dy = - (1/3.5) * 2 = -2/3.5 = -4/7 approx -0.57143
    
    ASSERT_EQ_APPROX(gy[{0}], -0.57143f, 1e-4, "Backward grad y index 0");
    
    // Index 1 (x=4, y=1):
    // inner = 3.25
    // d(inner)/dx = 1 - 1/16 = 0.9375
    // dz/dx = - (1/3.25) * 0.9375 = -0.9375/3.25 approx -0.28846
    
    ASSERT_EQ_APPROX(gx[{1}], -0.28846f, 1e-4, "Backward grad x index 1");
    
    // Index 1 (x=4, y=1):
    // d(inner)/dy = -1 + 2/4 = -0.5
    // dz/dy = - (1/3.25) * (-0.5) = 0.5/3.25 approx 0.15385
    
    ASSERT_EQ_APPROX(gy[{1}], 0.15385f, 1e-4, "Backward grad y index 1");
}

#endif

void test_chained_ops_more() {
    std::cout << "\n===[ test_ops.h - more chained tests ]===\n";

    // p, q, r are vectors of length 3
    Tensor p({3});
    p[{0}] = 1.0f; p[{1}] = 2.0f; p[{2}] = 3.0f;

    Tensor q({3});
    q[{0}] = 0.5f; q[{1}] = -1.0f; q[{2}] = 2.0f;

    Tensor r({3});
    r[{0}] = 2.0f; r[{1}] = 1.0f; r[{2}] = 0.5f;

    Tensor two({3});
    two.fill_inplace(2.0f);

    // Chain: u = p + q
    //        v = u * r
    //        w = v^2
    //        s = sum(w)
    //        t = log(s)
    //        y = -t

    Tensor u = p + q;
    Tensor v = u * r;
    Tensor w = v.pow(two);
    Tensor s = w.sum(0);
    Tensor t = s.log();
    Tensor y = -t;

    // Forward checks
    // s = 9 + 1 + 6.25 = 16.25 -> log = ~2.788095 -> y = -2.788095
    ASSERT_EQ_APPROX(y.item(), -2.788095f, 1e-5, "Forward scalar y");

    // Backward (do not retain graph)
    y.backward(false);

    Tensor gp = p.grad();
    Tensor gq = q.grad();
    Tensor gr = r.grad();

    // Expected (derived):
    // dy/dp_i = -2*(p_i+q_i)*r_i^2 / s
    // dy/dq_i = same as dy/dp_i
    // dy/dr_i = -2*(p_i+q_i)^2 * r_i / s  (equivalently -2*u_i^2*r_i/s)

    // Numeric expectations computed in test design
    ASSERT_EQ_APPROX(gp[{0}], -0.7384615f, 1e-5, "grad p[0]");
    ASSERT_EQ_APPROX(gq[{0}], -0.7384615f, 1e-5, "grad q[0]");
    ASSERT_EQ_APPROX(gr[{0}], -0.55384615f, 1e-5, "grad r[0]");

    ASSERT_EQ_APPROX(gp[{1}], -0.1230769f, 1e-6, "grad p[1]");
    ASSERT_EQ_APPROX(gq[{1}], -0.1230769f, 1e-6, "grad q[1]");
    ASSERT_EQ_APPROX(gr[{1}], -0.1230769f, 1e-6, "grad r[1]");

    ASSERT_EQ_APPROX(gp[{2}], -0.15384615f, 1e-6, "grad p[2]");
    ASSERT_EQ_APPROX(gq[{2}], -0.15384615f, 1e-6, "grad q[2]");
    ASSERT_EQ_APPROX(gr[{2}], -1.5384615f, 1e-5, "grad r[2]");
}
