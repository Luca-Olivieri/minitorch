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

    Tensor a = x + y;
    Tensor b = -x;
    Tensor c = x*y;
    Tensor d = c.pow(y);

    std::cout << d << '\n';
    std::cout << *(a.m_bw_op) << '\n';
    std::cout << *(b.m_bw_op) << '\n';
    std::cout << *(c.m_bw_op) << '\n';
    std::cout << *(d.m_bw_op) << '\n';
    // std::cout << y << '\n';

    return 0;
}
