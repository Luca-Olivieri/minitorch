#include <memory>

#include "modules.h"

namespace mt::nn {

    AbstractModule::AbstractModule()
        : m_parameters{}, m_modules{} {}
    
    void AbstractModule::requires_grad_(
        const bool requires_grad
    ) {
        for (auto& kv : m_modules) {
            kv.second.requires_grad = requires_grad;
        }
    }

    Module::Module(): AbstractModule() {}
    
    const AbstractModule& Module::register_module(
            const std::string& name,
            AbstractModule& module
    ) {
        m_modules.emplace(std::move(name), module);
        return module;
    }

    std::map<std::string, Tensor> AbstractModule::parameters() const {
        std::map<std::string, Tensor> out;

        // add this module's parameters first
        for (const auto& kv : m_parameters) {
            out.emplace(kv.first, kv.second);
        }

        // then recursively add child modules' parameters
        for (const auto& kv : m_modules) {
            const std::string& child_name = kv.first;
            const AbstractModule& child = kv.second;
            auto child_params = child.parameters();
            for (const auto& p : child_params) {
                std::string full_name = child_name;
                if (!p.first.empty()) full_name += "." + p.first;
                out.emplace(std::move(full_name), p.second);
            }
        }

        return out;
    }
}
