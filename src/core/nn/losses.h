#ifndef LOSSES_H
#define LOSSES_H

#include "modules.h"

namespace mt::nn {
    class ForwardLoss {
    public:
        virtual ~ForwardLoss() = default;
        
        virtual Tensor forward(
                const Tensor& inputs,
                const Tensor& targets
        ) const = 0;
    };

    class Loss : public Module, public ForwardLoss {
    public:
        virtual ~Loss() = default;
    };

    class MSELoss : public Loss {
    public:
        MSELoss();

        Tensor forward(
                const Tensor& inputs,
                const Tensor& targets
        ) const override;
    };

    class BCELossWithLogits : public Loss {
    public:
        BCELossWithLogits();

        Tensor forward(
                const Tensor& inputs,
                const Tensor& targets
        ) const override;
    };
    class CrossEntropyLoss : public Loss {
    public:
        CrossEntropyLoss();

        Tensor forward(
                const Tensor& inputs,
                const Tensor& targets
        ) const override;
    };
}

#endif
