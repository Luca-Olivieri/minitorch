#ifndef TEST_NN_H
#define TEST_NN_H

#include "src/core/tensors.h"
#include "src/core/nn/compute.h"
#include "src/core/nn/activations.h"
#include "tests/test_utils.h"

void test_linear_relu_forward_backward() {
    std::cout << "\n===[ test_nn: single Linear + ReLU ]===\n";

    // Linear initialized with weight=0.1 and bias=-0.5 in constructor
    mt::nn::Linear lin(2, 3, true);

    // Input chosen so that linear output is positive (so ReLU passes gradient)
    Tensor x({2});
    x[{0}] = 10.0f;
    x[{1}] = 0.0f;

    Tensor out = lin.forward(x);

    // Each output should be (10*0.1 + 0*0.1) + (-0.5) = 1.0 - 0.5 = 0.5
    for (size_t j = 0; j < 3; ++j) {
        ASSERT_EQ_APPROX(out[{j}], 0.5f, 1e-6, "Linear forward constant weights -> expected 0.5");
    }

    mt::nn::ReLU relu;
    Tensor y = relu.forward(out);
    Tensor loss = y.sum(0);
    loss.backward();

    // Input gradients: dL/dx_i = sum_j w_{i,j} * dL/dout_j
    // w_{i,j} == 0.1, out_features == 3, upstream dL/dout_j == 1 -> dL/dx_i = 3 * 0.1 = 0.3
    Tensor gx = x.grad();
    ASSERT_EQ_APPROX(gx[{0}], 0.3f, 1e-6, "Input grad[0] after Linear+ReLU backward");
    ASSERT_EQ_APPROX(gx[{1}], 0.3f, 1e-6, "Input grad[1] after Linear+ReLU backward");

    // Weight grads: dL/dw_{i,j} = x_i * dL/dout_j ; here dL/dout_j == 1
    for (size_t j = 0; j < 3; ++j) {
        ASSERT_EQ_APPROX(lin.m_weight.grad()[{0, j}], 10.0f, 1e-6, "Weight grad row0 == input0 * 1");
        ASSERT_EQ_APPROX(lin.m_weight.grad()[{1, j}], 0.0f, 1e-6, "Weight grad row1 == input1 * 1");
        ASSERT_EQ_APPROX(lin.m_bias.grad()[{j}], 1.0f, 1e-6, "Bias grad == 1 for each output");
    }
}

void test_two_layer_linear_relu_linear_backward() {
    std::cout << "\n===[ test_nn: two Linear layers with ReLU ]===\n";

    // l1: 2 -> 2, l2: 2 -> 1. Same deterministic init: weight=0.1, bias=-0.5
    mt::nn::Linear l1(2, 2, true);
    mt::nn::Linear l2(2, 1, true);
    mt::nn::ReLU relu;

    Tensor x({2});
    x[{0}] = 100.0f; // large so outputs stay positive through both layers
    x[{1}] = 0.0f;

    Tensor h = relu.forward(l1.forward(x));
    Tensor out = l2.forward(h);
    Tensor loss = out.sum(0); // scalar
    loss.backward();

    // l2 grads: dL/dw2_{i,0} = h_i * 1 -> h_i = l1_forward = 100*0.1 - 0.5 = 10 - 0.5 = 9.5
    for (size_t i = 0; i < 2; ++i) {
        ASSERT_EQ_APPROX(l2.m_weight.grad()[{i, 0}], 9.5f, 1e-6, "l2 weight grad == hidden activation");
    }
    ASSERT_EQ_APPROX(l2.m_bias.grad()[{0}], 1.0f, 1e-6, "l2 bias grad == 1");

    // Backprop into hidden: dL/dh_i = l2.weight_{i,0} * 1 -> weight is 0.1 for both
    // Then l1 weight grads: dL/dw1_{k,j} = x_k * dL/dout_j_of_l1 ; dL/dout_j_of_l1 == 0.1
    for (size_t j = 0; j < 2; ++j) {
        ASSERT_EQ_APPROX(l1.m_weight.grad()[{0, j}], 100.0f * 0.1f, 1e-6, "l1 weight grad row0 == input0 * 0.1");
        ASSERT_EQ_APPROX(l1.m_weight.grad()[{1, j}], 0.0f, 1e-6, "l1 weight grad row1 == input1 * 0.1");
        ASSERT_EQ_APPROX(l1.m_bias.grad()[{j}], 0.1f, 1e-6, "l1 bias grad == 0.1 for each hidden unit");
    }

    // Input grads: dL/dx_k = sum_j w1_{k,j} * dL/dout_j_of_l1
    // w1_{k,j} == 0.1, two outputs -> dL/dx_k = 2 * 0.1 * 0.1 = 0.02
    Tensor gx = x.grad();
    ASSERT_EQ_APPROX(gx[{0}], 0.02f, 1e-6, "Input grad[0] after two-layer backward");
    ASSERT_EQ_APPROX(gx[{1}], 0.02f, 1e-6, "Input grad[1] after two-layer backward");
}

#endif
