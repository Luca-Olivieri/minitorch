#include <iostream>

#include "core/tensors_impl.h"
#include "core/tensors.h"
#include "core/backward_ops.h"
#include "core/formatting.h"

int main()
{
    std::vector<int> vec = {1, 4, 6};

    std::cout << vec << '\n';

    Tensor x({2, 3});
    x.linspace(1, 6);
    
    Tensor y({2, 3});
    y.fill(2.0f);

    std::cout << x << '\n';
    std::cout << y << '\n';

    return 0;
}
