#include <iostream>

#include "core/ops.h"
#include "core/tensors.h"

int do_nothing(
    int x,
    int y
){
    std::cout << x + y << '\n';
    return 1;
}

int main()
{   
    Tensor t({2, 4});
    t.linspace(2, 4);
    // Tensor t({});

    // std::cout << t << '\n';
    // t.fill(3.0f);
    // std::cout << t.m_strides << '\n';
    // std::cout << t.item() << '\n';
    // std::cout << t[{0}] << '\n';

    
    std::cout << t;
    // t.reshape({4, 2});
    // std::cout << t;
    t.transpose(0, 1);
    std::cout << t;

    return 0;
}
