#include <iostream>

#include "core/tensor_storages.h"
#include "core/tensor_nodes.h"
#include "core/backward_ops.h"
#include "core/formatting.h"

int main()
{
    std::vector<int> vec = {1, 4, 6};

    std::cout << vec << '\n';

    TensorNode x({2, 3});
    x.linspace(1, 6);
    
    TensorNode y({2, 3});
    y.fill(2.0f);

    TensorNode lrs{x.m_value.m_shape};
    lrs.fill(1e-2f);

    for (size_t i {0}; i < 1000; i++) {
        TensorNode p = x.pow(y);
        TensorNode m = -x;
        TensorNode o = p + m;
        o.backward();
        x += (-lrs)* *(x.m_grad);
        o.zero_grad();
    }

    std::cout << x << '\n';

    return 0;
}
