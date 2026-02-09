#include <iostream>

#include "core/tensors.h"
#include "core/formatting.h"

Tensor forward(
    Tensor& inputs
) {
    Tensor y({inputs.shape()});
    y.fill_inplace(2.0f);

    return inputs.pow(y) - inputs;
}

int main()
{
    Tensor x {{3, 2}};

    Tensor lrs{x.shape()};
    lrs.fill_inplace(1e-2f);

    for (size_t i = 0; i < 1000; i++) {
        Tensor o {forward(x)};
        o.backward();
        x += -lrs * x.grad();
        o.zero_grad();
    }

    std::cout << x << '\n';

    return 0;
}
