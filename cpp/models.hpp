#pragma once

#include <torch/torch.h>

namespace nn = torch::nn;

template<class Encoder>
struct SupervisedGraphSageImpl : nn::Module {

  SupervisedGraphSageImpl(int64_t inum_classes,
                          Encoder ienc)
    : enc(ienc),
      weight(register_parameter("weight", torch::randn({inum_classes, enc->embed_dim})))
  { }


  torch::Tensor forward(torch::Tensor &nodes)
  {
    auto embeds = enc(nodes);
    auto scores = weight.mm(embeds);
    return scores.t();
  }


  Encoder enc;
  torch::Tensor weight;
};
