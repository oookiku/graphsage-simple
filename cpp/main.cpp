#include "encoders.hpp"
#include "utils.hpp"
#include <torch/torch.h>

namespace nn = torch::nn;


//
// 1st layer
//
using mean_agg_impl = MeanAggregatorImpl<nn::Embedding,
                                         false,
                                         false>;
TORCH_MODULE_IMPL(MeanAggregator0, mean_agg_impl);


using mean_enc_impl = EncoderImpl<MeanAggregator0,
                                  nn::Embedding,
                                  false,
                                  false>;
TORCH_MODULE_IMPL(MeanEncoder0, mean_enc_impl);


//
// 2nd layer
//
using mean_agg_impl1 = MeanAggregatorImpl<MeanEncoder0,
                                          false,
                                          false>;
TORCH_MODULE_IMPL(MeanAggregator1, mean_agg_impl1);

using mean_enc_impl1 = EncoderImpl<MeanAggregator1,
                                   MeanEncoder0,
                                   false,
                                   false>;
TORCH_MODULE_IMPL(MeanEncoder1, mean_enc_impl1);


int main()
{
  constexpr int64_t num_nodes = 2708;
  constexpr int64_t num_feats = 1433;
  constexpr int64_t num_embed = 128;

  auto [feat_data, labels, adj_list] = load_cora();

  auto features = nn::Embedding(num_nodes, num_feats);
  // features->weight = nn::Module::register_parameter("weight", feat_data.to(torch::kFloat32));

  auto agg1 = MeanAggregator0(features);

  auto enc1 = MeanEncoder0(features,
                           num_feats,
                           num_embed,
                           adj_list,
                           agg1);


  auto agg2 = MeanAggregator1(enc1);

  auto enc2 = MeanEncoder1(enc1,
                           enc1->embed_dim,
                           num_embed,
                           adj_list,
                           agg2);
}
