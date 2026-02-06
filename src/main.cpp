#include <iostream>

#include "core/tensors_storage.h"
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

    Tensor lrs{x.m_value.m_shape};
    lrs.fill(1e-2f);

    for (size_t i {0}; i < 1000; i++) {
        Tensor p = x.pow(y);
        Tensor m = -x;
        Tensor o = p + m;
        o.backward();
        x += (-lrs)* *(x.m_grad);
        o.zero_grad();
    }

    std::cout << x << '\n';

    return 0;
}
