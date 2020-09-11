#pragma once

#include <iostream>
#include <unordered_map>
#include <boost/range/adaptor/indexed.hpp>


std::tuple<torch::Tensor,
           torch::Tensor,
           matrix_int64>
load_cora()
{
  constexpr int64_t num_nodes = 2708;
  constexpr int64_t num_feats = 1433;
  auto feat_data = torch::zeros({num_nodes, num_feats}, torch::kFloat32);
  auto labels = torch::empty({num_nodes, 1}, torch::kInt64);
  std::unordered_map<int64_t, int64_t> node_map(num_nodes);
  std::vector<int64_t> label_map(num_nodes);

  //
  // loading node infomation & converting node id into idx of arrays 
  //
  std::string fname1 = "../../cora/cora.content";
  std::ifstream file1(fname1);
  if (!file1) {
    std::cout << "Could not open " << fname1 << std::endl;
  }
  else {
    std::cout << "Loading "  << fname1 << std::endl;
  }

  int64_t idx = 0, info_feats[num_feats+1];
  while (file1 >> info_feats[0]) {
    for (int64_t i = 1; i < num_feats+1; ++i) {
      int64_t u;
      file1 >> u;
      info_feats[i] = u;
    }

    for (int64_t i = 0; i < num_feats; ++i) {
      feat_data[idx][i] = static_cast<float>(info_feats[i+1]);
    }
 
    node_map[info_feats[0]] = idx;
     
    std::string v;
    file1 >> v;
    labels[idx] = static_cast<int64_t>(v.size());
    idx++;
  }


  //
  // loading edge infomation & creating adjacent list
  //
  matrix_int64 adj_lists(num_nodes);
  std::string fname2 = "../../cora/cora.cites";
  std::ifstream file2(fname2);
  if (!file2) { 
    std::cout << "Could not open " << fname2 << std::endl;
  }
  else {
    std::cout << "Loading " << fname2 << std::endl;
  }

  int64_t info_edge[2];
  while (file2 >> info_edge[0] >> info_edge[1]) {
    auto paper1 = node_map.at(info_edge[0]);
    auto paper2 = node_map.at(info_edge[1]);
    // std::cout << paper1 << " " << paper2 << std::endl;
    adj_lists[paper1].push_back(paper2);
    adj_lists[paper2].push_back(paper1);
  }


  return {feat_data, labels, adj_lists};
}
