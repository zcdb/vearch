/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This faiss source code is licensed under the MIT license.
 * https://github.com/facebookresearch/faiss/blob/master/LICENSE
 *
 *
 * The works below are modified based on faiss:
 * 1. Replace the static batch indexing with real time indexing
 * 2. Add the numeric field and bitmap filters in the process of searching
 *
 * Modified works copyright 2019 The Gamma Authors.
 *
 * The modified codes are licensed under the Apache License, Version 2.0 license
 * found in the LICENSE file in the root directory of this source tree.
 *
 */

#pragma once

#pragma once

#include <faiss/IndexIVFFlat.h>
#include <faiss/utils/distances.h>
#include <faiss/MetricType.h>

#include "index/realtime/realtime_invert_index.h"
#include "util/status.h"
#include "gamma_index_flat.h"

namespace vearch {

using idx_t = faiss::idx_t;

template <faiss::MetricType metric, class C, bool use_sel>
struct GammaIVFFlatScanner : faiss::InvertedListScanner {
  size_t d;
  const RetrievalContext* retrieval_context_;
  GammaIVFFlatScanner(size_t d, bool store_pairs, const faiss::IDSelector* sel, 
    const RetrievalContext* retrieval_context) : 
    InvertedListScanner(store_pairs, sel), d(d), retrieval_context_(retrieval_context) {
      keep_max = is_similarity_metric(metric);
    }

  const float *xi;
  void set_query(const float *query) override { this->xi = query; }

  idx_t list_no;
  void set_list(idx_t list_no, float /* coarse_dis */) override {
    this->list_no = list_no;
  }

  float distance_to_code(const uint8_t *code) const override {
    const float *yj = (float *)code;
    float dis = metric == faiss::METRIC_INNER_PRODUCT
                    ? faiss::fvec_inner_product(xi, yj, d)
                    : faiss::fvec_L2sqr(xi, yj, d);
    return dis;
  }

  size_t scan_codes(size_t list_size, const uint8_t *codes, const idx_t *ids,
                    float *simi, idx_t *idxi, size_t k) const override {
    const float *list_vecs = (const float *)codes;
    size_t nup = 0;
    for (size_t j = 0; j < list_size; j++) {
      if (ids[j] & realtime::kDelIdxMask) {
        continue;
      }
      idx_t vid = ids[j] & realtime::kRecoverIdxMask;
      if (!retrieval_context_->IsValid(vid)) {
        continue;
      }
      const float *yj = list_vecs + d * j;
      float dis = metric == faiss::METRIC_INNER_PRODUCT
                      ? faiss::fvec_inner_product(xi, yj, d)
                      : faiss::fvec_L2sqr(xi, yj, d);
      if (retrieval_context_->IsSimilarScoreValid(dis) &&
        C::cmp(simi[0], dis)) {
        int64_t id = store_pairs ? faiss::lo_build(list_no, j) : ids[j];
        faiss::heap_replace_top<C>(k, simi, idxi, dis, id);
        nup++;
      }
    }
    return nup;
  }
};

template <bool use_sel>
faiss::InvertedListScanner* get_InvertedListScanner1(
        const faiss::IndexIVFFlat* ivf,
        bool store_pairs,
        const faiss::IDSelector* sel,
        const RetrievalContext* retrieval_context,
        faiss::MetricType metric_type) {
    if (metric_type == faiss::METRIC_INNER_PRODUCT) {
        return new GammaIVFFlatScanner<
                faiss::METRIC_INNER_PRODUCT,
                faiss::CMin<float, int64_t>,
                use_sel>(ivf->d, store_pairs, sel, retrieval_context);
    } else if (metric_type == faiss::METRIC_L2) {
        return new GammaIVFFlatScanner<faiss::METRIC_L2, faiss::CMax<float, int64_t>, use_sel>(
                ivf->d, store_pairs, sel, retrieval_context);
    } else {
        FAISS_THROW_MSG("metric type not supported");
    }
}

class IVFFlatRetrievalParameters : public RetrievalParameters {
 public:
  IVFFlatRetrievalParameters() : RetrievalParameters() {
    parallel_on_queries_ = true;
    nprobe_ = -1;
  }

  IVFFlatRetrievalParameters(bool parallel_on_queries, int nprobe,
                             enum DistanceComputeType type)
      : RetrievalParameters() {
    parallel_on_queries_ = parallel_on_queries;
    nprobe_ = nprobe;
    distance_compute_type_ = type;
  }

  IVFFlatRetrievalParameters(int nprobe, enum DistanceComputeType type) {
    parallel_on_queries_ = true;
    nprobe_ = nprobe;
    distance_compute_type_ = type;
  }

  virtual ~IVFFlatRetrievalParameters() {}

  int Nprobe() { return nprobe_; }

  void SetNprobe(int nprobe) { nprobe_ = nprobe; }

  bool ParallelOnQueries() { return parallel_on_queries_; }

  void SetParallelOnQueries(bool parallel_on_queries) {
    parallel_on_queries_ = parallel_on_queries;
  }

 protected:
  // parallelize over queries or ivf lists
  bool parallel_on_queries_;
  int nprobe_;
};

struct GammaIndexIVFFlat : faiss::IndexIVFFlat, public GammaFLATIndex {
  GammaIndexIVFFlat();
  virtual ~GammaIndexIVFFlat();

  void search_preassigned(RetrievalContext *retrieval_context, idx_t n,
                          const float *x, int k, const idx_t *keys,
                          const float *coarse_dis, float *distances,
                          idx_t *labels, int nprobe, bool store_pairs);

  Status Init(const std::string &model_parameters,
              int training_threshold) override;
  RetrievalParameters *Parse(const std::string &parameters) override;
  int Indexing() override;
  bool Add(int n, const uint8_t *vec) override;
  int Update(const std::vector<int64_t> &ids,
             const std::vector<const uint8_t *> &vecs) override;
  int Delete(const std::vector<int64_t> &ids) override;
  // int AddRTVecsToIndex() override;

  int Search(RetrievalContext *retrieval_context, int n, const uint8_t *x,
             int k, float *distances, idx_t *labels) override;

  long GetTotalMemBytes() override { return 0; };

  Status Dump(const std::string &dir) override;
  Status Load(const std::string &dir, int64_t &load_num) override;

  void train(int64_t n, const float *x) override {
    faiss::IndexIVFFlat::train(n, x);
  }

  void Describe() override;

 private:
  faiss::InvertedListScanner *GetGammaInvertedListScanner(
    bool store_pairs,
    const faiss::IDSelector* sel,
    const RetrievalContext * retrieval_context,
    faiss::MetricType metric_type) const;

 protected:
  int indexed_vec_count_;

 private:
  realtime::RTInvertIndex *rt_invert_index_ptr_;
  uint64_t updated_num_;
#ifdef PERFORMANCE_TESTING
  int add_count_;
#endif
  DistanceComputeType metric_type_;
};

}  // namespace vearch
