#ifndef MODULES_H
#define MODULES_H

#include <memory>
#include <map>

#include "src/core/tensors.h"

namespace mt::nn
{
    class AbstractModule {
    public:
        bool requires_grad { true };

        virtual ~AbstractModule() = default;
        
        virtual std::shared_ptr<AbstractModule> register_module(
            std::string name,
            std::shared_ptr<AbstractModule> module
        ) = 0;

        virtual Tensor forward(
            const Tensor& input
        ) const = 0;

        void requires_grad_(
            const bool requires_grad = true
        );

    protected:
        std::map<std::string, std::shared_ptr<AbstractModule>> m_modules;
    };

    class Module : public AbstractModule {
    public:
        std::shared_ptr<AbstractModule> register_module(
                std::string name,
                std::shared_ptr<AbstractModule> module
        ) override;
    };
}

#endif
