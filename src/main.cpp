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
    Tensor t({2, 3, 5});
    // Tensor t({});

    // std::cout << t << '\n';
    // t.fill(3.0f);
    // std::cout << t.m_strides << '\n';
    // std::cout << t.item() << '\n';
    // std::cout << t[{0}] << '\n';

    std::cout << t.get_slice(0).m_shape;

    return 0;
}
