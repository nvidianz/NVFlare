/**
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>
#include <algorithm>   // for copy_n, transform
#include <cstring>     // for memcpy
#include <memory>      // for shared_ptr
#include <stdexcept>   // for invalid_argument
#include <string_view> // for string_view
#include <vector>      // for vector

#include "nvflare_plugin.h"
#include "data_set_ids.h"
#include "dam.h"       // for DamEncoder

namespace nvflare {

NvflarePlugin::NvflarePlugin(
    std::vector<std::pair<std::string_view, std::string_view>> const &args): BasePlugin(args){
}

void NvflarePlugin::EncryptGPairs(float const *in_gpair, std::size_t n_in,
                                  std::uint8_t **out_gpair,
                                  std::size_t *n_out) {
  std::vector<double> pairs(n_in);
  std::copy_n(in_gpair, n_in, pairs.begin());
  DamEncoder encoder(kDataSetGHPairs);
  encoder.AddFloatArray(pairs);
  encrypted_gpairs_ = encoder.Finish(*n_out);
  if (!out_gpair) {
    throw std::invalid_argument{"Invalid pointer to output gpair."};
  }
  *out_gpair = encrypted_gpairs_.data();
  *n_out = encrypted_gpairs_.size();
}

void NvflarePlugin::SyncEncryptedGPairs(std::uint8_t const *in_gpair,
                                        std::size_t n_bytes,
                                        std::uint8_t const **out_gpair,
                                        std::size_t *out_n_bytes) {
  *out_n_bytes = n_bytes;
  *out_gpair = in_gpair;
}

void NvflarePlugin::ResetHistContext(std::uint32_t const *cutptrs,
                                     std::size_t cutptr_len,
                                     std::int32_t const *bin_idx,
                                     std::size_t n_idx) {
  // fixme: this doesn't have to be copied multiple times.
  this->cut_ptrs_.resize(cutptr_len);
  std::copy_n(cutptrs, cutptr_len, cut_ptrs_.begin());
  this->bin_idx_.resize(n_idx);
  std::copy_n(bin_idx, n_idx, this->bin_idx_.begin());
}

void NvflarePlugin::BuildEncryptedHistVert(std::size_t const **ridx,
                                           std::size_t const *sizes,
                                           std::int32_t const *nidx,
                                           std::size_t len,
                                           std::uint8_t** out_hist,
                                           std::size_t* out_len) {
  std::int64_t data_set_id;
  if (!feature_sent_) {
    data_set_id = kDataSetAggregationWithFeatures;
    feature_sent_ = true;
  } else {
    data_set_id = kDataSetAggregation;
  }

  DamEncoder encoder(data_set_id);

  // Add cuts pointers
  std::vector<int64_t> cuts_vec(cut_ptrs_.cbegin(), cut_ptrs_.cend());
  encoder.AddIntArray(cuts_vec);

  auto num_features = cut_ptrs_.size() - 1;
  auto num_samples = bin_idx_.size() / num_features;

  if (data_set_id == kDataSetAggregationWithFeatures) {
    if (features_.empty()) { // when is it not empty?
      for (std::size_t f = 0; f < num_features; f++) {
        auto slot = bin_idx_[f];
        if (slot >= 0) {
          // what happens if it's missing?
          features_.push_back(f);
        }
      }
    }
    encoder.AddIntArray(features_);

    std::vector<int64_t> bins;
    for (int i = 0; i < num_samples; i++) {
      for (auto f : features_) {
        auto index = f + i * num_features;
        if (index > bin_idx_.size()) {
          throw std::out_of_range{"Index is out of range: " +
                                  std::to_string(index)};
        }
        auto slot = bin_idx_[index];
        bins.push_back(slot);
      }
    }
    encoder.AddIntArray(bins);
  }

  // Add nodes to build
  std::vector<int64_t> node_vec(len);
  std::copy_n(nidx, len, node_vec.begin());
  encoder.AddIntArray(node_vec);

  // For each node, get the row_id/slot pair
  for (std::size_t i = 0; i < len; ++i) {
    std::vector<int64_t> rows(sizes[i]);
    std::copy_n(ridx[i], sizes[i], rows.begin());
    encoder.AddIntArray(rows);
  }

  std::size_t n{0};
  encrypted_hist_ = encoder.Finish(n);

  *out_hist = encrypted_hist_.data();
  *out_len = encrypted_hist_.size();
}

void NvflarePlugin::SyncEncryptedHistVert(std::uint8_t *buffer,
                                          std::size_t buf_size, double **out,
                                          std::size_t *out_len) {
  auto remaining = buf_size;
  char *pointer = reinterpret_cast<char *>(buffer);

  // The buffer is concatenated by AllGather. It may contain multiple DAM
  // buffers
  std::vector<double> &result = hist_;
  result.clear();
  auto max_slot = cut_ptrs_.back();
  auto array_size = 2 * max_slot * sizeof(double);
  // A new histogram array?
  double *slots = static_cast<double *>(malloc(array_size));
  while (remaining > kPrefixLen) {
    DamDecoder decoder(reinterpret_cast<uint8_t *>(pointer), remaining);
    if (!decoder.IsValid()) {
      std::cout << "Not DAM encoded buffer ignored at offset: "
                << static_cast<int>(
                       (pointer - reinterpret_cast<char *>(buffer)))
                << std::endl;
      break;
    }
    auto size = decoder.Size();
    auto node_list = decoder.DecodeIntArray();
    for (auto node : node_list) {
      std::memset(slots, 0, array_size);
      auto feature_list = decoder.DecodeIntArray();
      // Convert per-feature histo to a flat one
      for (auto f : feature_list) {
        auto base = cut_ptrs_[f]; // cut pointer for the current feature
        auto bins = decoder.DecodeFloatArray();
        auto n = bins.size() / 2;
        for (int i = 0; i < n; i++) {
          auto index = base + i;
          // [Q] Build local histogram? Why does it need to be built here?
          slots[2 * index] += bins[2 * i];
          slots[2 * index + 1] += bins[2 * i + 1];
        }
      }
      result.insert(result.end(), slots, slots + 2 * max_slot);
    }
    remaining -= size;
    pointer += size;
  }
  free(slots);

  *out_len = result.size();
  *out = result.data();
}

void NvflarePlugin::BuildEncryptedHistHori(double const *in_histogram,
                                           std::size_t len,
                                           std::uint8_t **out_hist,
                                           std::size_t *out_len) {
  DamEncoder encoder(kDataSetHistograms);
  std::vector<double> copy(in_histogram, in_histogram + len);
  encoder.AddFloatArray(copy);

  std::size_t size{0};
  this->encrypted_hist_ = encoder.Finish(size);

  *out_hist = this->encrypted_hist_.data();
  *out_len = this->encrypted_hist_.size();
}

void NvflarePlugin::SyncEncryptedHistHori(std::uint8_t const *buffer,
                                          std::size_t len, double **out_hist,
                                          std::size_t *out_len) {
  DamDecoder decoder(reinterpret_cast<uint8_t const *>(buffer), len);
  if (!decoder.IsValid()) {
    std::cout << "Not DAM encoded buffer, ignored" << std::endl;
  }

  if (decoder.GetDataSetId() != kDataSetHistogramResult) {
    throw std::runtime_error{"Invalid dataset: " +
                             std::to_string(decoder.GetDataSetId())};
  }
  this->hist_ = decoder.DecodeFloatArray();
  *out_hist = this->hist_.data();
  *out_len = this->hist_.size();
}
} // namespace nvflare
