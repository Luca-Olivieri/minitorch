#include <iostream>
#include "src/core/tensors.h"
#include "src/core/nn/losses.h"

int main() {
    try {
        std::cout << "start\n";
        Tensor inputs({2,3});
        inputs[{0,0}] = 1.0f; inputs[{0,1}] = 2.0f; inputs[{0,2}] = 3.0f;
        inputs[{1,0}] = 0.5f; inputs[{1,1}] = -1.0f; inputs[{1,2}] = 0.0f;
        std::cout << "inputs set\n";

        Tensor targets({2,3});
        targets[{0,2}] = 1.0f; targets[{1,0}] = 1.0f;
        targets.detach_inplace();
        std::cout << "targets set\n";

        mt::nn::CrossEntropyLoss loss;
        std::cout << "loss constructed\n";

        Tensor out = loss.forward(inputs, targets);
        std::cout << "forward done, out shape size=" << out.shape().size() << "\n";
        std::cout << "out dims:";
        for (auto d : out.shape()) std::cout << d << ",";
        std::cout << "\n";

        out.backward();
        std::cout << "backward done\n";

        Tensor g = inputs.grad();
        std::cout << "grad shape size=" << g.shape().size() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cout << "caught exception: " << e.what() << "\n";
        return 1;
    }
}
