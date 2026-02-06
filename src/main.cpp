#include <iostream>

#include "core/tensors.h"
#include "core/tensor_storages.h"
#include "core/tensor_nodes.h"
#include "core/backward_ops.h"
#include "core/formatting.h"

int main()
{
    // tensor_nodes_loop();

    Tensor x({2, 3});
    x.linspace(1, 6);

    Tensor y({2, 3});
    y.fill(2.0f);

    Tensor s = x + y;

    std::cout << s << '\n';

    return 0;
}
