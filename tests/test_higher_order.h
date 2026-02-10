#ifndef TEST_HIGHER_ORDER_H
#define TEST_HIGHER_ORDER_H

#include <iostream>
#include <cmath>
#include "test_utils.h"
#include "src/core/tensors.h"

void test_higher_order_derivatives() {
    
    std::cout << "\n===[ test_higher_order.h ]===\n";
    
    // Test 1: Second derivative of x^2 => 2
    {
        Tensor x({1});
        x.fill_inplace(3.0f); // x = 3
        
        Tensor y = x * x; // y = x^2 = 9
        
        // create_graph=true to enable higher order grads
        y.backward(true);
        
        Tensor grad_x = x.grad(); // dy/dx = 2x = 6
        ASSERT_EQ_APPROX(grad_x.item(), 6.0f, 1e-4, "First derivative dy/dx");
        
        // We accumulate into grad_x, so it's not strictly necessary to zero here if we just look at the new contribution to the gradient of the gradient.
        // However, x.grad() holds the gradient w.r.t x.
        // grad_x holds the Tensor that represents dy/dx.
        // We want d(dy/dx)/dx.
        // grad_x is effectively a function of x.
        // So we backward from grad_x.
        
        // If we want to check x.grad() after the second backward, it will accumulate.
        // Initial x.grad() was 6.
        // Second backward adds d(grad_x)/dx = 2.
        // Total x.grad() should be 8.
        
        grad_x.backward(); 
        
        Tensor grad2_x = x.grad();
        ASSERT_EQ_APPROX(grad2_x.item(), 8.0f, 1e-4, "Second derivative accumulated (6 + 2)");
        
        // If we isolated the second derivative:
        // x.zero_grad();
        // grad_x.backward();
        // ASSERT_EQ(x.grad().item(), 2.0f);
    }
    
    // Test 2: Third derivative of x^3 => 6
    {
        Tensor x({1});
        x.fill_inplace(2.0f); // x = 2
        
        Tensor pow3({1});
        pow3.fill_inplace(3.0f);
        Tensor y = x.pow(pow3); // y = x^3 = 8
        
        y.backward(true); 
        Tensor grad_1 = x.grad(); // dy/dx = 3x^2 = 3*4 = 12
        ASSERT_EQ_APPROX(grad_1.item(), 12.0f, 1e-4, "First derivative of x^3");
        
        x.zero_grad(); // Clear gradient for clean accumulation check if desired, though usually we accumulate.
        // Let's clear to verify isolation.
        
        grad_1.backward(true); // create_graph=true for next level
        Tensor grad_2 = x.grad(); // d2y/dx2 = 6x = 12
        ASSERT_EQ_APPROX(grad_2.item(), 12.0f, 1e-4, "Second derivative of x^3");
        
        x.zero_grad();
        
        grad_2.backward(); // create_graph not needed for final value
        Tensor grad_3 = x.grad(); // d3y/dx3 = 6
        ASSERT_EQ_APPROX(grad_3.item(), 6.0f, 1e-4, "Third derivative of x^3");
    }
    
    // Test 3: Second derivative of sin(x) => -sin(x) ? 
    // We don't have sin implemented yet, assume polynomial P(x) = x^4 + 2x^2
    {
        Tensor x({1});
        x.fill_inplace(2.0f);
        
        Tensor p4({1}); p4.fill_inplace(4.0f);
        Tensor p2({1}); p2.fill_inplace(2.0f);
        
        Tensor term1 = x.pow(p4); // x^4
        
        Tensor coeff2({1}); coeff2.fill_inplace(2.0f);
        Tensor term2 = x * x * coeff2; // 2x^2
        Tensor y = term1 + term2; // x^4 + 2x^2
        // y' = 4x^3 + 4x
        // y'' = 12x^2 + 4
        
        // At x=2:
        // y' = 4(8) + 8 = 40
        // y'' = 12(4) + 4 = 52
        
        y.backward(true);
        ASSERT_EQ_APPROX(x.grad().item(), 40.0f, 1e-4, "First derivative of poly");
        
        Tensor grad_1 = x.grad();
        x.zero_grad(); // Clear to check exact value of second derivative
        
        grad_1.backward();
        ASSERT_EQ_APPROX(x.grad().item(), 52.0f, 1e-4, "Second derivative of poly");
    }

    // Test 4: Multivariate 2nd derivative (Hessian vector product implicit)
    // f(x, y) = x^2 * y
    // df/dx = 2xy
    // d2f/dxdy = 2x
    {
        Tensor x({1}); x.fill_inplace(3.0f);
        Tensor y({1}); y.fill_inplace(4.0f);
        
        Tensor u = x * x;
        Tensor f = u * y; // x^2 * y = 9 * 4 = 36
        
        f.backward(true);
        
        // df/dx = 2xy = 2*3*4 = 24
        // df/dy = x^2 = 9
        
        ASSERT_EQ_APPROX(x.grad().item(), 24.0f, 1e-4, "df/dx");
        ASSERT_EQ_APPROX(y.grad().item(), 9.0f, 1e-4, "df/dy");
        
        // Check u.grad
        // u.grad should be y = 4
        // partial w.r.t u in f = u*y is y.
        std::cout << "u.grad value: " << u.grad().item() << std::endl;
        
        Tensor df_dx = x.grad();
        
        std::cout << "df_dx value: " << df_dx.item() << std::endl;
        
        x.zero_grad();
        y.zero_grad(); // clears y.grad (which was u=9).
        u.zero_grad(); // clear u.grad (which was y=4). 
                       // Wait, if df_dx depends on u.grad, and I zero u.grad, do I break df_dx graph?
                       // u.grad IS a node in df_dx graph.
                       // u.zero_grad() replaces u.m_node->m_grad with a new separate tensor.
                       // The OLD u.grad comes from f.backward() step.
                       // df_dx refers to the OLD u.grad.
                       // So u.zero_grad() is SAFE for df_dx.
        
        std::cout << "After zero_grad, y.grad: " << (y.grad().item()) << std::endl;
        
        // Differentiate df/dx w.r.t y
        df_dx.backward();
        
        std::cout << "x.grad (d(df/dx)/dx): " << x.grad().item() << std::endl;
        std::cout << "y.grad (d(df/dx)/dy): " << y.grad().item() << std::endl;
        
        // d(2xy)/dx = 2y = 8 (accumulates into x via backward)
        // d(2xy)/dy = 2x = 6 (accumulates into y via backward)
        
        ASSERT_EQ_APPROX(x.grad().item(), 8.0f, 1e-4, "d(df/dx)/dx");
        
        ASSERT_EQ_APPROX(y.grad().item(), 6.0f, 1e-4, "d(df/dx)/dy");
    }
    
    // Test 5: Simple Mixed Second Derivative (x*y)
    {
        Tensor x({1}); x.fill_inplace(3.0f);
        Tensor y({1}); y.fill_inplace(5.0f);
        
        Tensor f = x * y;
        f.backward(true);
        
        Tensor grad_x = x.grad(); // should be y = 5
        ASSERT_EQ_APPROX(grad_x.item(), 5.0f, 1e-4, "Simple xy grad_x");
        
        x.zero_grad();
        y.zero_grad();
        
        grad_x.backward();
        
        // d(y)/dy = 1
        ASSERT_EQ_APPROX(y.grad().item(), 1.0f, 1e-4, "Simple xy mixed partial");
    }
}

#endif
