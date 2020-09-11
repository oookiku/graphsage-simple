#include "encoders.hpp"
#include "utils.hpp"
#include "models.hpp"
#include <torch/torch.h>
#include <random>
#include <iterator>


namespace nn = torch::nn;
namespace F  = torch::nn::functional;

// 1st layer
using mean_agg_impl = MeanAggregatorImpl<nn::Embedding,
                                         false,
                                         false>;
TORCH_MODULE_IMPL(MeanAggregator0, mean_agg_impl);


using mean_enc_impl = EncoderImpl<MeanAggregator0,
                                  nn::Embedding,
                                  false,
                                  false>;
TORCH_MODULE_IMPL(MeanEncoder0, mean_enc_impl);


// 2nd layer
using mean_agg_impl1 = MeanAggregatorImpl<MeanEncoder0,
                                          false,
                                          false>;
TORCH_MODULE_IMPL(MeanAggregator1, mean_agg_impl1);

using mean_enc_impl1 = EncoderImpl<MeanAggregator1,
                                   MeanEncoder0,
                                   false,
                                   false>;
TORCH_MODULE_IMPL(MeanEncoder1, mean_enc_impl1);


// model
using sgraphsage_impl = SupervisedGraphSageImpl<MeanEncoder1>;
TORCH_MODULE_IMPL(SupervisedGraphSage, sgraphsage_impl);


int main()
{
  constexpr int64_t num_nodes = 2708;
  constexpr int64_t num_feats = 1433;
  constexpr int64_t num_embed = 128;
  constexpr int64_t num_class = 7;

  auto [feat_data, labels, adj_list] = load_cora();

  auto features = nn::Embedding(num_nodes, num_feats);
  features->weight = feat_data.to(torch::kFloat32);


  // creating node indicaters
  std::vector<int64_t> rand_indices(num_nodes);
  for (int64_t i = 0; i < num_nodes; ++i) {
    rand_indices[i] = i;
  }
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::shuffle(rand_indices.begin(),
               rand_indices.end(),
               engine);

  std::vector<int64_t> test, val, train;
  auto begin = rand_indices.begin();
  auto end   = rand_indices.end();
  std::copy(begin,
            begin+999,
            std::back_inserter(test));
  std::copy(begin+1000,
            begin+1499,
            std::back_inserter(val));
  std::copy(begin+1500,
            end,
            std::back_inserter(train));


  // aggregator & encoder
  auto agg1 = MeanAggregator0(features);

  auto enc1 = MeanEncoder0(features,
                           num_feats,
                           num_embed,
                           adj_list,
                           std::move(agg1));


  auto agg2 = MeanAggregator1(enc1);

  auto enc2 = MeanEncoder1(std::move(enc1),
                           enc1->embed_dim,
                           num_embed,
                           adj_list,
                           std::move(agg2));


  // model & optimizer
  auto graphsage = SupervisedGraphSage(num_class,
                                       std::move(enc2));

  auto optimizer = torch::optim::SGD(graphsage->parameters(), 
                                     /*lr=*/0.7);


  // training loop
  for (int64_t batch = 0; batch < 100; ++batch) {
    std::cout << "batch = " << batch << std::endl;

    std::vector<int64_t> batch_nodes_tmp;
    std::copy(train.begin(),
              train.begin()+255,
              std::back_inserter(batch_nodes_tmp));


    auto batch_nodes = torch::from_blob(batch_nodes_tmp.data(),
                                        batch_nodes_tmp.size(),
                                        torch::TensorOptions().dtype(torch::kInt64));

    auto train_labels = torch::zeros(batch_nodes_tmp.size(), torch::kInt64);
    for (auto&& e : batch_nodes_tmp | boost::adaptors::indexed()) {
      train_labels[e.index()] = labels[e.value()].item<int64_t>();
    }

    std::shuffle(train.begin(),
                 train.end(),
                 engine);


    optimizer.zero_grad();
    auto output = graphsage(batch_nodes);
    auto loss = F::cross_entropy(output, train_labels.squeeze());
    loss.backward();
    optimizer.step();
  }

}
