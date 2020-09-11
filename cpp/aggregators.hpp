#pragma once

#include <torch/torch.h>
#include <boost/range/adaptor/indexed.hpp>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <type_traits>

namespace nn = torch::nn;

template<class Embedding = nn::Embedding,
         bool CUDA = false,
         bool GCN = false>
struct MeanAggregatorImpl : nn::Module {

  MeanAggregatorImpl(const Embedding ifeatures)
    : features(ifeatures)
  { }

  torch::Tensor forward(const torch::Tensor &nodes,
                        std::vector<std::vector<int64_t>> &to_neighs,
                        int64_t num_sample=10)
  {
    //
    // sampling neighbor nodes by using adj_list
    //
    std::vector<std::vector<int64_t>> samp_neighs;
    if constexpr (GCN) {
      samp_neighs.resize(nodes.size(0));
      for (auto&& p : to_neighs | boost::adaptors::indexed()) {
        // sample all neighbors and myself
        samp_neighs[p.index()] = p.value();
        samp_neighs[p.index()].push_back(nodes[p.index()].item<int64_t>());
      }
    }
    else {
      if (num_sample != 0) {
        samp_neighs.resize(nodes.size(0));
        for (auto&& p : to_neighs | boost::adaptors::indexed()) {
          // sample (num_sample)-nodes randomly
          // note: num_sample=min(p.value.size(), num_sample)
          std::random_device seed_gen;
          std::mt19937 engine {seed_gen()};

          std::vector<int64_t> sample;
          std::sample(p.value().begin(),
                      p.value().end(),
                      std::back_inserter(sample),
                      num_sample,
                      engine);
          samp_neighs[p.index()] = sample;
        }
      }
      else {
        // sample all neighbors
        samp_neighs = to_neighs;
      }
    }


    //
    // creating unique nodes list
    //
    // 2d vector to 1d set
    std::set<int64_t> tmp;
    for (auto&& p : samp_neighs) {
      tmp.insert(p.begin(), p.end());
    }
    // 1d set to 1d vector
    std::vector<int64_t> unique_nodes_list(tmp.begin(), tmp.end());

    std::unordered_map<int64_t, int64_t> unique_nodes(unique_nodes_list.size());
    for (auto&& p : unique_nodes_list | boost::adaptors::indexed()) {
      unique_nodes[p.value()] = p.index();
    }

    //
    // creating a mask matrix which outputs mean features
    //
    const int64_t samp_neighs_size = samp_neighs.size();
    const int64_t unique_nodes_size = unique_nodes.size();

    std::vector<int64_t> column_indices, row_indices;
    for (auto&& samp_neigh : samp_neighs | boost::adaptors::indexed()) {
      for (auto&& n : samp_neigh.value()) {
        column_indices.push_back(unique_nodes.at(n));
        row_indices.push_back(samp_neigh.index());
        // std::cout << n << ":" << unique_nodes.at(n) << " ";
      }
      // std::cout << std::endl;
    }

    auto mask = torch::zeros({samp_neighs_size, unique_nodes_size});
    for (int64_t i = 0; i < column_indices.size(); ++i) {
      const int64_t column = column_indices[i];
      const int64_t row = row_indices[i];
      mask[row][column] = 1;
    }
    if constexpr (CUDA) mask.cuda();
    auto num_neigh = mask.sum(/*dim=*/1,
                              /*keepdim=*/true);
    mask = mask.div(num_neigh);


    //
    // extracting features of the unique nodes
    //
    torch::Tensor embed_matrix;
    if constexpr (CUDA) {
      embed_matrix = features(torch::from_blob(unique_nodes_list.data(), 
                                               unique_nodes_list.size(),
                                               torch::TensorOptions().dtype(torch::kInt64))
                              .cuda());
    }
    else {
      embed_matrix = features(torch::from_blob(unique_nodes_list.data(), 
                                               unique_nodes_list.size(),
                                               torch::TensorOptions().dtype(torch::kInt64)));
    }


    //
    // calculating mean features
    //
    torch::Tensor to_feats;
    if constexpr (std::is_same_v<Embedding, nn::Embedding>) {
      to_feats = mask.mm(embed_matrix); 
    }
    else {
      to_feats = mask.mm(embed_matrix.t()); 
    }
    return to_feats;
  }

  Embedding features;
};
