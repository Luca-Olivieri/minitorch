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
    
    Tensor exp({2, 3});
    exp.fill(2.0f);
    
    // Tensor c({2, 3, 4});
    // c.fill(2.0f);
    // Tensor t({});

    // std::cout << t << '\n';
    // t.fill(3.0f);
    // std::cout << t.m_strides << '\n';
    // std::cout << t.item() << '\n';
    // std::cout << t[{0}] << '\n';

    // Tensor d = t * c;

    // std::cout << *(d.m_grad_fn) << '\n';
    
    // d.backward();
    
    // std::cout << d.m_flat_grad << '\n';

    // std::cout << t.m_flat_grad << '\n';

    // Tensor z = -x;
    // std::cout << z << '\n';

    float lr = 1e-2f;

    for (int i = 0; i < 1000; i++) {
        x.reset_grads();
        Tensor p = x.pow(exp);
        Tensor m = -x;
        Tensor y = p + m;
        y.backward();
        for (size_t j = 0; j < x.m_numel; j++) {
            size_t x_logic_idx = x.get_flat_index_from_logical(j);
            x.m_flat_data[x_logic_idx] -= lr*x.m_flat_grad[x_logic_idx];
        }
        
        std::cout << x.m_flat_data << '\n';
        std::cout << y.m_flat_data << '\n' << '\n';
    }

    return 0;
}
