#include <memory>

#include "modules.h"

namespace mt::nn {
    
    void AbstractModule::requires_grad_(
        const bool requires_grad
    ) {
        for (auto& kv : m_modules) {
            kv.second->requires_grad = requires_grad;
        }
    }
    
    std::shared_ptr<AbstractModule> Module::register_module(
            std::string name,
            std::shared_ptr<AbstractModule> module
    ) {
        m_modules[name] = module;
        return module;
    }
}
