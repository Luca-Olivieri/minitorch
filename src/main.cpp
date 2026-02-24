#include <iostream>
#include <map>
#include <chrono>

#include "core/reproducibility.h"
#include "core/formatting.h"
#include "core/tensors.h"
#include "core/tensor_nodes.h"
#include "core/tensor_storages.h"
#include "src/core/nn/modules.h"
#include "src/core/nn/compute.h"
#include "src/core/nn/activations.h"
#include "src/core/nn/losses.h"
#include "src/core/nn/optimizers.h"
#include "src/data/datasets.h"
#include "src/data/dataloaders.h"
#include "src/io/csv.h"

namespace nn = mt::nn;

void try_covertype() {

    class CovertypeDataset: public mt::data::ClassificationDataset {
    public:
        CovertypeDataset(
            const std::string& path,
            size_t limit = 0
        ): m_csv_reader(path) {

            // m_len = m_csv_reader.size();
            m_len = limit ? limit : m_csv_reader.size();
        }

        std::tuple<Tensor, Tensor> getitem(
                size_t index
        ) {
            std::string line = m_csv_reader.read_line(index);
            std::stringstream ss(line);
            std::string value;

            Tensor input({54});
            // discard 'index' column
            std::getline(ss, value, ',');

            for (size_t col {0}; col < 54; ++col) {
                std::getline(ss, value, ',');
                float f = std::stof(value);
                input[{col}] = f;
            }

            Tensor gt{std::vector<size_t>()};
            std::getline(ss, value, ',');
            gt.item() = std::stof(value) - 1;

            return {input, gt};
        }

        size_t len() const {
            return m_len;
        }
    private:
        io::CSVReader m_csv_reader;
    };

    class CovertypeClassifier: public nn::Module, nn::Forward1 {
    public:
        nn::Linear lin1;
        nn::ReLU relu;
        nn::Linear lin2;
        nn::Linear lin3;

        CovertypeClassifier():
            lin1(54, 100),
            relu{},
            lin2(100, 100),
            lin3(100, 7) {
                register_module("lin1", lin1);
                register_module("lin2", lin2);
                register_module("lin3", lin3);
                reset_parameters();
            }

        Tensor forward(
                const Tensor& inputs
        ) const {
            Tensor y1 = lin1.forward(inputs);
            Tensor y2 = relu.forward(y1);
            Tensor y3 = lin2.forward(y2);
            Tensor y4 = relu.forward(y3);
            Tensor prs = lin3.forward(y4);
            return prs;
        }

        void reset_parameters() {
            nn::kaiming_uniform_inplace(lin1.m_weight, get_rng());
            lin1.m_bias.fill_inplace(0.0f);
            nn::kaiming_uniform_inplace(lin2.m_weight, get_rng());
            lin2.m_bias.fill_inplace(0.0f);
            nn::kaiming_uniform_inplace(lin3.m_weight, get_rng());
            lin3.m_bias.fill_inplace(0.0f);
        }

        Tensor evaluate(
            mt::data::DataLoader<Tensor, Tensor>& dl,
            nn::Loss& criterion
        ) {
            float curr_loss = 0;
            float curr_sample_count = 0;

            for (size_t step { 0 }; step < dl.size(); ++step) {
                auto [inputs, gts] = dl.get_batch(step);
                
                Tensor prs_oh = forward(inputs);

                Tensor gts_oh = gts.one_hot(prs_oh.shape()[1]);
            
                Tensor loss = criterion.forward(prs_oh, gts_oh);

                curr_loss += loss.item()*(static_cast<float>(inputs.shape()[0]));
                curr_sample_count += static_cast<float>(inputs.shape()[0]);
            }
            float total_loss = curr_loss / curr_sample_count;
            return Tensor({}, total_loss);
        }
    };

    auto START = std::chrono::high_resolution_clock::now();

    const size_t limit = 100;

    CovertypeDataset train_ds(config["covertype_train_path"], limit);
    CovertypeDataset val_ds(config["covertype_val_path"], limit);

    std::cout << std::format("Dataset set up (took {} s)",
        static_cast<double>((std::chrono::high_resolution_clock::now() - START).count())/1e9
    ) << '\n';

    START = std::chrono::high_resolution_clock::now();

    mt::data::DataLoader<Tensor, Tensor> train_dl(train_ds, 4, true, get_rng());
    mt::data::DataLoader<Tensor, Tensor> val_dl(val_ds, 4, false, get_rng());

    std::cout << std::format("Dataloaders set up (took {} s)",
        static_cast<double>((std::chrono::high_resolution_clock::now() - START).count())/1e9
    ) << '\n';

    START = std::chrono::high_resolution_clock::now();

    CovertypeClassifier model{};

    std::cout << std::format("Model set up (took {} s)",
        static_cast<double>((std::chrono::high_resolution_clock::now() - START).count())/1e9
    ) << '\n';

    START = std::chrono::high_resolution_clock::now();

    nn::CrossEntropyLoss criterion{};
    auto params = model.parameters();
    SGD optimizer(params, config["base_lr"]);

    std::cout << std::format("Criterion and optimizer set up (took {} s)",
        static_cast<double>((std::chrono::high_resolution_clock::now() - START).count())/1e9
    ) << '\n';

    START = std::chrono::high_resolution_clock::now();

    auto init_val_loss = model.evaluate(val_dl, criterion);

    std::cout << std::format("Initial loss: {} (took {} s)",
        init_val_loss.item(),
        static_cast<double>((std::chrono::high_resolution_clock::now() - START).count())/1e9
    ) << '\n';

    const size_t num_epochs = 10;

    for (size_t epoch { 0 }; epoch < num_epochs; ++epoch) {

        START = std::chrono::high_resolution_clock::now();

        for (size_t step { 0 }; step < train_dl.size(); ++step) {
            auto [inputs, gts] = train_dl.get_batch(step);
            
            Tensor prs_oh = model.forward(inputs);

            Tensor gts_oh = gts.one_hot(prs_oh.shape()[1]);

            Tensor loss = criterion.forward(prs_oh, gts_oh);
            loss.backward();

            // if (step % 10 == 0) {
            //     std::cout << loss << '\n';
            // }
            
            optimizer.step();
            optimizer.zero_grad();
        }

        std::cout << std::format("[Epoch {}/{}] Training done: (took {} s)",
            epoch+1,
            num_epochs,
            static_cast<double>((std::chrono::high_resolution_clock::now() - START).count())/1e9
        ) << '\n';

        START = std::chrono::high_resolution_clock::now();
        
        auto epoch_val_loss = model.evaluate(val_dl, criterion);
        
        std::cout << std::format("[Epoch {}/{}] val. loss: {} (took {} s)",
            epoch+1,
            num_epochs,
            epoch_val_loss.item(),
            static_cast<double>((std::chrono::high_resolution_clock::now() - START).count())/1e9
        ) << '\n';

        train_dl.reshuffle();
    }

    START = std::chrono::high_resolution_clock::now();

    auto final_train_loss = model.evaluate(train_dl, criterion);

    std::cout << std::format("Final training loss: {} (took {} s)",
        final_train_loss.item(),
        static_cast<double>((std::chrono::high_resolution_clock::now() - START).count())/1e9
    ) << '\n';
}

int main()
{
    try_covertype();

    return 0;
}
