#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <map>
#include <string>

#include "src/core/tensors.h"

class Optimizer {
public:
    std::map<std::string, Tensor>& m_parameters;
    const float m_base_lr;

    virtual ~Optimizer() = default;

    Optimizer(
        std::map<std::string, Tensor>& parameters,
        float base_lr
    );

    virtual void step() = 0;

    void zero_grad();
};

class SGD: public Optimizer {
public:
    using Optimizer::Optimizer;

    void step() override;
};

#endif
