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
    // Tensor t({2, 3, 4});
    Tensor t({});

    // TODO: implement a .h file for each test file and then include all of them in test.cpp in which you run them.

    // TODO: implement slicing.

    std::cout << t << '\n';
    t.fill(3.0f);
    std::cout << t.m_strides << '\n';
    std::cout << t.item() << '\n';
    std::cout << t[{0}] << '\n';

    return 0;
}
