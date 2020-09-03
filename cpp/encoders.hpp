#pragma once

#include <torch/torch.h>
#include "aggregators.hpp"
#include "typedef.hpp"

namespace nn = torch::nn;
namespace F  = torch::nn::functional;

template<class Aggregator>
struct EncoderImpl : nn::Module {

  EncoderImpl(nn::Embedding ifeatures,
              int64_t ifeature_dim,
              int64_t iembed_dim,
              matrix_int64 &iadj_list,
              Aggregator iaggregator,
              int64_t inum_sample=10,
              bool igcn=false,
              bool icuda=false)
    : features(ifeatures),
      feat_dim(ifeature_dim),
      embed_dim(iembed_dim),
      adj_list(iadj_list),
      aggregator(iaggregator),
      num_sample(inum_sample),
      gcn(igcn),
      cuda(icuda),
      weight(register_parameter("weight", torch::randn({embed_dim, 
                                                        (gcn)? feat_dim : 2*feat_dim})))
  { }


  torch::Tensor forward(std::vector<int64_t> &nodes) 
  {
    matrix_int64 sorted_adj_list(nodes.size());
    for (int64_t i = 0; i < nodes.size(); ++i) {
      sorted_adj_list[i] = adj_list[nodes[i]];
    }
    auto neigh_feats = aggregator->forward(nodes,
                                           sorted_adj_list,
                                           num_sample);

    torch::Tensor self_feats, combined;
    if (!gcn) {
      if (cuda) {
        self_feats = features(torch::from_blob(nodes.data(), 
                                               nodes.size())
                              .to(torch::kInt64)
                              .cuda());
      }
      else {
        self_feats = features(torch::from_blob(nodes.data(),
                                               nodes.size())
                              .to(torch::kInt64));
      }
      combined = torch::cat({self_feats, neigh_feats}, 1);
    }
    else {
      combined = neigh_feats;
    }

    return F::relu(weight.mm(combined.t()));
  }


  int64_t feat_dim, embed_dim, num_sample;
  bool gcn, cuda;
  matrix_int64 adj_list;
  Aggregator aggregator;
  nn::Embedding features;
  torch::Tensor weight; 
};

TORCH_MODULE_IMPL(MeanEncoder, EncoderImpl<MeanAggregator>);

// TORCH_MODULE(MaxpoolAggregator, EncoderImple<MaxpoolAggregator>);
