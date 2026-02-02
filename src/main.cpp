#include <iostream>

#include "core/tensors_impl.h"
#include "core/tensors.h"
#include "core/autograd.h"

int do_nothing(
    int x,
    int y
){
    std::cout << x + y << '\n';
    return 1;
}

int main()
{   
    Tensor x({2, 3});
    x.linspace(1, 6);
    
    Tensor y({2, 3});
    y.fill(2.0f);

    std::cout << x << '\n';
    std::cout << y << '\n';

    return 0;
}
