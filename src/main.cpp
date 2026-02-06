#include <iostream>

#include "core/tensors.h"
#include "core/tensor_storages.h"
#include "core/tensor_nodes.h"
#include "core/backward_ops.h"
#include "core/formatting.h"

Tensor forward(
    Tensor& inputs
) {
    Tensor y({2, 3});
    y.fill(2.0f);
    
    Tensor p = inputs.pow(y);
    Tensor m = -inputs;
    Tensor o = p + m;
    return o;
}

int main()
{
    // tensor_nodes_loop();

    Tensor x({2, 3});
    x.linspace(1, 6);

    Tensor lrs{x.m_node->m_value.m_shape};
    lrs.fill(1e-2f);

    for (size_t i = 0; i < 1000; i++) {
        Tensor o {forward(x)};
        o.backward();
        *(x.m_node) += *(*((-lrs).m_node) * *(x.m_node->m_grad));
        o.zero_grad();
    }

    std::cout << x << '\n';

    return 0;
}
