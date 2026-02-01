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
    Tensor x({2, 3});
    x.linspace(1, 6);
    
    Tensor y({2, 3});
    y.fill(2.0f);

    Tensor z = x*y;

    std::cout << z << '\n';
    
    z.backward();
    
    std::cout << x << '\n';
    std::cout << y << '\n';
    std::cout << *(x.m_grad) << '\n';

    // for (int i = 0; i < 1000; i++) {
    //     x.reset_grads();
    //     Tensor p = x.pow(exp);
    //     Tensor m = -x;
    //     Tensor y = p + m;
    //     y.backward();
    //     for (size_t j = 0; j < x.m_numel; j++) {
    //         size_t x_logic_idx = x.get_flat_index_from_logical(j);
    //         x.m_flat_data[x_logic_idx] -= lr*x.m_flat_grad[x_logic_idx];
    //     }
        
    //     std::cout << x.m_flat_data << '\n';
    //     std::cout << y.m_flat_data << '\n' << '\n';
    // }

    return 0;
}
