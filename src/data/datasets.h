#ifndef DATASETS_H
#define DATASETS_H

#include <tuple>

#include "src/core/tensors.h"

namespace mt::data {
    template <typename... Rs>
    class Dataset {
    public:
        virtual ~Dataset() = default;

        virtual std::tuple<Rs...> getitem(
                size_t index
        ) = 0;
    
        virtual size_t len() const = 0;
    protected:
        size_t m_len { 0 };
    };
    
    class ClassificationDataset: public Dataset<Tensor, Tensor> {};
}

#endif
