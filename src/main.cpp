#include <iostream>

#include "core/tensors.h"
#include "core/tensor_nodes.h"
#include "core/tensor_storages.h"
#include "core/formatting.h"

Tensor forward_1st(
    Tensor& inputs
) {
    Tensor exp_2 { inputs.shape(), 2.0f };

    return inputs.pow(exp_2) - inputs;
}

void optimize_1st_deriv() {
    Tensor x {{3, 2}};
    x.linspace_inplace(0.1f, 0.9f);

    Tensor lrs { x.shape(), 1e-2f };

    for (size_t i = 0; i < 500; i++) {
        Tensor o {forward_1st(x)};
        o.backward();
        x += -lrs * x.grad();
        o.zero_grad();
    }

    std::cout << x << '\n';
}

Tensor forward_2nd(
    Tensor& inputs
) {
    Tensor exp_2 { inputs.shape(), 2.0f };
    
    Tensor exp_3{ inputs.shape(), 3.0f };

    return inputs.pow(exp_3) - inputs.pow(exp_2);
}

void optimize_2nd_deriv() {
    Tensor x {{3, 2}};
    x.linspace_inplace(0.1f, 0.9f);

    Tensor lrs { x.shape(), 1e-2f };

    for (size_t i = 0; i < 500; i++) {
        Tensor o {forward_2nd(x)};
        o.backward(true);
        
        Tensor grad_x { x.grad() };
        x.zero_grad();
        grad_x.backward();
        
        x += -lrs * x.grad();
        x.zero_grad();
    }

    std::cout << x << '\n';
}

int main()
{
    // optimize_1st_deriv();
    // optimize_2nd_deriv();

    // Tensor a { Tensor::linspace({2, 2, 3}, 0.0f, 11.0f) };

    // std::cout << a << '\n';
    // std::cout << a.m_node->m_storage.m_strides << '\n';

    TensorStorage a { {2, 3, 4} };
    a.linspace_inplace(0.0f, 23.0f);

    std::cout << a << '\n';

    TensorStorage s = TensorStorage::s_sum(a, 1);
    std::cout << s << '\n';
    
    return 0;
}
