#include <iostream>
#include <map>

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
#include "src/io/csv.h"

namespace nn = mt::nn;

Tensor forward_1st(
    Tensor& inputs
) {
    Tensor exp_2 { inputs.shape(), 2.0f, false };

    return inputs.pow(exp_2) - inputs;
}

void optimize_1st_deriv() {
    Tensor x = Tensor::linspace({2, 3}, -1.0f, 2.0f);

    Tensor lrs { x.shape(), 1e-2f };

    for (size_t i = 0; i < 500; i++) {
        Tensor o {forward_1st(x)};
        o.backward();
        x += -lrs * x.grad();
        x.zero_grad();
    }

    std::cout << x << '\n';
}

void try_nn() {
    Tensor x = Tensor::linspace({12}, 0.1f, 0.9f);
    nn::Linear lin1(12, 18);
    nn::kaiming_normal_inplace(lin1.m_weight, get_rng());
    
    nn::ReLU relu{};
    Tensor y = lin1.forward(x);
    Tensor z = relu.forward(y);
    z.backward();
    std::cout << x.grad() << '\n';
    std::cout << lin1.m_weight.grad() << '\n';
    std::cout << lin1.m_bias.grad() << '\n';
}

class IrisDataset: public data::ClassificationDataset {
public:
    IrisDataset(
        const std::string& path
    ): m_csv_reader(path) {

        m_len = 0;

        flower_to_class.emplace("Iris-setosa", 0.0f);
        flower_to_class.emplace("Iris-versicolor", 1.0f);
        flower_to_class.emplace("Iris-virginica", 2.0f);
    }

    std::tuple<Tensor, Tensor> getitem(
            size_t index
    ) {
        std::string line = m_csv_reader.read_line(index);
        std::stringstream ss(line);
        std::string value;

        Tensor input({4});
        // discard 'Id' column
        std::getline(ss, value, ',');

        std::getline(ss, value, ',');
        input[{0}] = std::stof(value);

        std::getline(ss, value, ',');
        input[{1}] = std::stof(value);
        
        std::getline(ss, value, ',');
        input[{2}] = std::stof(value);
        
        std::getline(ss, value, ',');
        input[{3}] = std::stof(value);

        Tensor gt{std::vector<size_t>()};
        std::getline(ss, value, ',');
        gt.item() = flower_to_class.at(value);

        return {input, gt};
    }

    size_t len() const {
        return m_len;
    }
private:
    io::CSVReader m_csv_reader;
    std::map<std::string, float> flower_to_class;
};

void try_xor() {
    
    Tensor x({4, 2}, 0.0f, false);
    x[{0, 0}] = 0.0f; x[{0, 1}] = 0.0f;
    x[{1, 0}] = 0.0f; x[{1, 1}] = 1.0f;
    x[{2, 0}] = 1.0f; x[{2, 1}] = 0.0f;
    x[{3, 0}] = 1.0f; x[{3, 1}] = 1.0f;
    
    Tensor gts({4}, 0.0f, false);
    gts[{0}] = 0.0f;
    gts[{1}] = 1.0f;
    gts[{2}] = 1.0f;
    gts[{3}] = 0.0f;

    class FFN: public nn::Module, nn::Forward1 {
    public:
        nn::Linear lin1;
        nn::ReLU relu;
        nn::Linear lin2;
        nn::Linear lin3;

        FFN():
            lin1(2, 4),
            relu{},
            lin2(4, 3),
            lin3(3, 1) {
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
            Tensor out = lin3.forward(y4);
            Tensor prs = out.squeeze(1);
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
    };

    FFN ffn{};

    nn::BCELossWithLogits criterion{};
    auto params = ffn.parameters();
    std::cout << params << '\n';
    SGD optimizer(params, config["base_lr"]);

    {
        Tensor prs_ = ffn.forward(x);
        std::cout << prs_ << '\n';
        
        std::cout << criterion.forward(prs_, gts);
    }

    for (size_t i = 0; i < 2000; i++) {
        Tensor prs = ffn.forward(x);

        Tensor loss = criterion.forward(prs, gts);
        loss.backward();
        
        optimizer.step();
        optimizer.zero_grad();
    }

    {
        Tensor prs_ = ffn.forward(x);
        std::cout << prs_ << '\n';
        
        std::cout << criterion.forward(prs_, gts);
    }
}

int main()
{
    // optimize_1st_deriv();
    // optimize_2nd_deriv();
    // try_nn();
    // try_xor();

    std::string iris_path = "data/Iris.csv";

    IrisDataset ds(iris_path);

    std::cout << ds.getitem(150);

    return 0;
}
