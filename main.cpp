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

    std::cout << t[0, 0, 1] << '\n';
    t.fill(3.0f);
    std::cout << t << '\n';

    return 0;
}
