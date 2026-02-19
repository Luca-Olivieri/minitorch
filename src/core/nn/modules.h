#ifndef MODULES_H
#define MODULES_H

#include <memory>
#include <map>
#include <string>

#include "src/core/tensors.h"

namespace mt::nn
{
    class AbstractModule {
    public:
        bool requires_grad { true };

        AbstractModule();

        virtual ~AbstractModule() = default;
        
        virtual std::map<std::string, Tensor> parameters() const;
        virtual const AbstractModule& register_module(
                const std::string& name,
                AbstractModule& module
        ) = 0;

        void requires_grad_(
            const bool requires_grad = true
        );

    protected:
        std::map<std::string, Tensor> m_parameters;
        std::map<std::string, AbstractModule&> m_modules;
    };

    class Module : public AbstractModule {
    public:
        
        Module();

        const AbstractModule& register_module(
                const std::string& name,
                AbstractModule& module
        ) override;
    };

    class Forward1 {
    public:
        virtual ~Forward1() = default;
        
        virtual Tensor forward(
            const Tensor& inputs
        ) const = 0;
    };
}

#endif
