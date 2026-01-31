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
    Tensor t({2, 3, 4});
    t.linspace(1, 24);
    // Tensor t({});

    // std::cout << t << '\n';
    // t.fill(3.0f);
    // std::cout << t.m_strides << '\n';
    // std::cout << t.item() << '\n';
    // std::cout << t[{0}] << '\n';

    Tensor c({});
    c.item() = 2.0f;

    t * c;
    t.pow(c);

    std::cout << t;

    return 0;
}
