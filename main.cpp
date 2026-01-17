#include <iostream>

#include "core/ops.h"

int do_nothing(
    int x,
    int y
){
    std::cout << x + y << '\n';
    return 1;
}

int main()
{   
    Node x(3.0f);
    Node y(-1.0f);

    // Addition add(x, y);
    // Multiplication mult(add, z);

    // add.forward();
    // mult.forward();

    // mult.backward();

    // std::cout << x << '\n';
    // std::cout << y << '\n';
    // std::cout << add << '\n';
    // std::cout << z << '\n';
    // std::cout << mult << '\n';
    
    Squaration square(x);
    Multiplication mult(x, y);
    Addition add(square, mult);

    const float lr { 1e-1f };

    for (int i {0}; i < 100; i++) {

        square.forward();
        mult.forward();
        add.forward();
        add.backward();

        if (i % 10) {
            std::cout << std::format("Iteration {}", i+1) << '\n';
            std::cout << x << '\n';
            std::cout << add << '\n';
            std::cout << "---" << '\n';
        }

        x.m_value -= lr*x.m_grad;
        
        add.reset_grads();
    }
    
    return EXIT_SUCCESS;
}
