#pragma once

#include <torch/torch.h>

namespace nn = torch::nn;

template<class Encoder>
struct SupervisedGraphSageImpl : nn::Module {

  SupervisedGraphSageImpl(int64_t inum_classes,
                          Encoder ienc)
    : enc(ienc),
      weight(register_parameter("weight", torch::randn({inum_classes, enc->embed_dim}))),
      xent(nn::CrossEntropyLoss())
  { }


  torch::Tensor forward(torch::Tensor &nodes)
  {
    auto embeds = enc(nodes);
    auto scores = weight.mm(embeds);
    return scores.t();
  }

  auto loss(torch::Tensor &nodes,
            torch::Tensor &labels)
  {
    auto scores = forward(nodes);
    return xent(scores, labels.squeeze());
  }


  Encoder enc;
  torch::Tensor weight;
  nn::CrossEntropyLoss xent;
};
