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
    
    
    // std::cout << *(y.m_grad) << '\n';

    Tensor lrs{x.m_shape};
    lrs.fill( 1e-2f);

    for (int i = 0; i < 1000; i++) {
        x.reset_grads();
        Tensor p = x.pow(y);
        Tensor m = -x;
        Tensor o = p + m;
        
        o.backward();
        
        Tensor a = -lrs;
        Tensor b = a*(*x.m_grad);
        Tensor temp = x + b;
        x = temp;
        x.m_grad_fn = nullptr;

        std::cout << o << '\n';
        std::cout << x << '\n' << '\n';
    }

    return 0;
}
