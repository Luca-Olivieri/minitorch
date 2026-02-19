#include <iostream>

#include "core/reproducibility.h"
#include "core/formatting.h"
#include "core/tensors.h"
#include "core/tensor_nodes.h"
#include "core/tensor_storages.h"
#include "src/core/nn/modules.h"
#include "src/core/nn/compute.h"
#include "src/core/nn/activations.h"
#include "src/core/nn/losses.h"

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
    mt::nn::kaiming_normal_inplace(lin1.m_weight, get_rng());
    
    mt::nn::ReLU relu{};
    Tensor y = lin1.forward(x);
    Tensor z = relu.forward(y);
    z.backward();
    std::cout << x.grad() << '\n';
    std::cout << lin1.m_weight.grad() << '\n';
    std::cout << lin1.m_bias.grad() << '\n';
}

void try_xor() {
    
    Tensor x({4, 2}, 0.0f, false);
    x[{0, 0}] = 0.0f; x[{0, 1}] = 0.0f;
    x[{1, 0}] = 0.0f; x[{1, 1}] = 1.0f;
    x[{2, 0}] = 1.0f; x[{2, 1}] = 0.0f;
    x[{3, 0}] = 1.0f; x[{3, 1}] = 1.0f;
    
    Tensor gts({4}, 0.0f, false);
    gts[{0}] = 0.0f;
    gts[{1}] = 1.0f;
    gts[{2}] = 1.0f;
    gts[{3}] = 0.0f;

    mt::nn::Linear lin1(2, 4);
    mt::nn::kaiming_normal_inplace(lin1.m_weight, get_rng());
    mt::nn::ReLU relu{};
    mt::nn::Linear lin2(4, 3);
    mt::nn::kaiming_normal_inplace(lin2.m_weight, get_rng());
    mt::nn::Linear lin3(3, 1);
    mt::nn::xavier_normal_inplace(lin3.m_weight, get_rng());

    mt::nn::BCELossWithLogits criterion{};

    float base_lr = 1e-1f;

    Tensor lrs_lin1_weight(lin1.m_weight.shape(), base_lr, false);
    Tensor lrs_lin1_bias(lin1.m_bias.shape(), base_lr, false);
    Tensor lrs_lin2_weight(lin2.m_weight.shape(), base_lr, false);
    Tensor lrs_lin2_bias(lin2.m_bias.shape(), base_lr, false);
    Tensor lrs_lin3_weight(lin3.m_weight.shape(), base_lr, false);
    Tensor lrs_lin3_bias(lin3.m_bias.shape(), base_lr, false);

    {
        Tensor y1_ = lin1.forward(x);
        Tensor y2_ = relu.forward(y1_);
        Tensor y3_ = lin2.forward(y2_);
        Tensor y4_ = relu.forward(y3_);
        Tensor out_ = lin3.forward(y4_);
        Tensor prs_ = out_.squeeze(1);
        std::cout << prs_ << '\n';
        
        std::cout << criterion.forward(prs_, gts);
    }

    for (size_t i = 0; i < 2000; i++) {
        Tensor y1 = lin1.forward(x);
        Tensor y2 = relu.forward(y1);
        Tensor y3 = lin2.forward(y2);
        Tensor y4 = relu.forward(y3);
        Tensor out = lin3.forward(y4);
        Tensor prs = out.squeeze(1);

        Tensor loss = criterion.forward(prs, gts);
        loss.backward();

        lin1.m_weight += -lrs_lin1_weight * lin1.m_weight.grad();
        lin1.m_bias += -lrs_lin1_bias * lin1.m_bias.grad();
        lin2.m_weight += -lrs_lin2_weight * lin2.m_weight.grad();
        lin2.m_bias += -lrs_lin2_bias * lin2.m_bias.grad();
        lin3.m_weight += -lrs_lin3_weight * lin3.m_weight.grad();
        lin3.m_bias += -lrs_lin3_bias * lin3.m_bias.grad();
        lin1.m_weight.zero_grad();
        lin1.m_bias.zero_grad();
        lin2.m_weight.zero_grad();
        lin2.m_bias.zero_grad();
        lin3.m_weight.zero_grad();
        lin3.m_bias.zero_grad();
    }

    {
        Tensor y1_ = lin1.forward(x);
        Tensor y2_ = relu.forward(y1_);
        Tensor y3_ = lin2.forward(y2_);
        Tensor y4_ = relu.forward(y3_);
        Tensor out_ = lin3.forward(y4_);
        Tensor prs_ = out_.squeeze(1);
        std::cout << prs_ << '\n';
        
        std::cout << criterion.forward(prs_, gts);
    }
}

int main()
{
    // optimize_1st_deriv();
    // optimize_2nd_deriv();
    // try_nn();
    try_xor();
    
    return 0;
}
