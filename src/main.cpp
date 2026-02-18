#include <iostream>

#include "core/formatting.h"
#include "core/tensors.h"
#include "core/tensor_nodes.h"
#include "core/tensor_storages.h"
#include "src/core/nn/compute.h"
#include "src/core/nn/activations.h"

Tensor forward_1st(
    Tensor& inputs
) {
    Tensor exp_2 { inputs.shape(), 2.0f, false };

    return inputs.pow(exp_2) - inputs;
}

void optimize_1st_deriv() {
    Tensor x = Tensor::linspace({2, 3}, -1.0f, 2.0f);

    Tensor lrs { x.shape(), 1e-2f };

    for (size_t i = 0; i < 500; i++) {
        Tensor o {forward_1st(x)};
        o.backward();
        x += -lrs * x.grad();
        x.zero_grad();
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
        o.backward();
        
        Tensor grad_x { x.grad() };
        x.zero_grad();
        grad_x.backward();
        
        x += -lrs * x.grad();
        x.zero_grad();
    }

    std::cout << x << '\n';
}

void try_nn() {
    Tensor x = Tensor::linspace({12}, 0.1f, 0.9f);
    mt::nn::Linear lin1(12, 18);
    mt::nn::ReLU relu{};
    Tensor y = lin1.forward(x);
    Tensor z = relu.forward(y);
    z.backward();
    std::cout << x.grad() << '\n';
    std::cout << lin1.m_weight.grad() << '\n';
    std::cout << lin1.m_bias.grad() << '\n';
}

int main()
{
    // optimize_1st_deriv();
    // optimize_2nd_deriv();
    try_nn();
    
    return 0;
}
