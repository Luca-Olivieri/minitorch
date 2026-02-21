#ifndef DATALOADERS_H
#define DATALOADERS_H

#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>

#include "src/core/tensors.h"
#include "src/data/datasets.h"

namespace mt::data {

    template <typename... Rs>
    class DataLoader {
    public:
        mt::data::Dataset<Rs...>& m_dataset;
        const size_t m_batch_size;
        size_t m_num_batches;
        const bool m_shuffle;

        std::vector<size_t> m_indices;
        std::mt19937 m_rng;

        DataLoader(
                mt::data::Dataset<Rs...>& dataset,
                size_t batch_size,
                bool shuffle,
                std::mt19937&& rng
        ):
            m_dataset { dataset },
            m_batch_size { batch_size },
            m_shuffle { shuffle },
            m_rng { rng } {

            m_num_batches = static_cast<size_t>(
                std::ceil(static_cast<float>(m_dataset.len()) / static_cast<float>(batch_size))
            );

            m_indices.resize(m_dataset.len());
            std::iota(m_indices.begin(), m_indices.end(), 0);
            if (m_shuffle) {
                std::shuffle(m_indices.begin(), m_indices.end(), m_rng);
            }
        }

        // Reshuffle indices when shuffle mode is enabled.
        void reshuffle() {
            if (!m_shuffle) return;
            std::shuffle(m_indices.begin(), m_indices.end(), m_rng);
        }

        // Return the batch at `index` as a tuple of stacked tensors.
        std::tuple<Rs...> get_batch(size_t index) const {
            const size_t start = index * m_batch_size;
            const size_t end = std::min(start + m_batch_size, m_dataset.len());

            std::tuple<std::vector<Rs>...> buffers;

            auto append_sample = [&buffers](const std::tuple<Rs...>& sample) {
                append_sample_impl(buffers, sample, std::index_sequence_for<Rs...>{});
            };

            for (size_t i = start; i < end; ++i) {
                const size_t ds_idx = m_shuffle ? m_indices[i] : i;
                auto sample = m_dataset.getitem(ds_idx);
                append_sample(sample);
            }

            return build_batch_from_buffers(buffers, std::index_sequence_for<Rs...>{});
        }

    private:
        template <size_t... Is>
        static void append_sample_impl(
            std::tuple<std::vector<Rs>...>& buffers,
            const std::tuple<Rs...>& sample,
            std::index_sequence<Is...>
        ) {
            (std::get<Is>(buffers).push_back(std::get<Is>(sample)), ...);
        }

        template <size_t... Is>
        static std::tuple<Rs...> build_batch_from_buffers(
            std::tuple<std::vector<Rs>...>& buffers,
            std::index_sequence<Is...>
        ) {
            return std::make_tuple(mt::stack(std::get<Is>(buffers))...);
        }
    };
}

#endif
