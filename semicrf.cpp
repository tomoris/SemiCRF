#include "semicrf.hpp"
#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include <cfloat>
#include <cmath>
#include <map>
#include <algorithm>

using labels = std::pair<int, std::pair<int, int>>;

semiCRF::semiCRF(const unsigned int w_i_s, const int t_i_s, const std::map<std::string, unsigned int> &word2index)
{
  word_index_size = w_i_s;
  tag_index_size = t_i_s;
  feature_size_onetag = (6 * word_index_size) + tag_index_size;   //chunk_b^e, w_b, ..., w_e, w_b-1, w_e+1, tag_t-1
  feature_size = feature_size_onetag * tag_index_size;
  weight.resize(feature_size, 0.1);

  a2 = tag_index_size;
  a1 = MAXIMUM_LENGTH * a2;

  ai4 = tag_index_size;
  ai3 = MAXIMUM_LENGTH * ai4;
  ai2 = tag_index_size * ai3;
  ai1 = MAXIMUM_LENGTH * ai2;
  w2i = word2index;
}

//あとでunserializeを呼び出す必要がある
semiCRF::semiCRF(FILE *fp)
{
  std::cout << "loading model" << '\n';
  unserialize(fp);
}

std::vector<unsigned int> semiCRF::return_feature_indexes_for_grad_item_at_tkzjr_main(const std::vector<std::string> &words, const int t, const int k, const int z) {
  std::vector<unsigned int> v;
  std::string chunk = "";
  for(int i = t-k+1; i <= t; i++) {
    chunk += words[i];
  }
  auto chunk_itr = w2i.find(chunk);
  if (chunk_itr != w2i.end()) {
    unsigned int feature_chunk_index = (z * feature_size_onetag) + w2i[chunk];
    v.push_back(feature_chunk_index);
  }

  for(int i = t-k+1; i <= t; i++) {
    unsigned int feature_chunk_bow_index = (z * feature_size_onetag) + word_index_size + w2i[words[i]];
    v.push_back(feature_chunk_bow_index);
  }


  //w_b-1
  if (t-k >= 0) {
    unsigned int feature_w_index = (z * feature_size_onetag) + (2 * word_index_size) + w2i[words[t-k]];
    v.push_back(feature_w_index);
  }
  //w_e+1
  if (t+1 < words.size()) {
    unsigned int feature_w_index = (z * feature_size_onetag) + (3 * word_index_size) + w2i[words[t+1]];
    v.push_back(feature_w_index);
  }

  /*
  //w_b-2
  if (t-k-1 >= 0) {
    unsigned int feature_w_index = (z * feature_size_onetag) + (4 * word_index_size) + w2i[words[t-k-1]];
    v.push_back(feature_w_index);
  }
  //w_e+2
  if (t+2 < words.size()) {
    unsigned int feature_w_index = (z * feature_size_onetag) + (5 * word_index_size) + w2i[words[t+2]];
    v.push_back(feature_w_index);
   }
  */

  return v;
}

unsigned int semiCRF::return_feature_index_for_grad_item_at_tkzjr_tag_trainsition(const int z, const int r) {
    unsigned int feature_transition_tag_index = ((z+1) * feature_size_onetag) - tag_index_size + r;
  return feature_transition_tag_index;
}

void semiCRF::create_feature_vec_at_tkzjr(const std::vector<std::string> &words,
                                            const int t, const int k, const int z,
                                            const int j, const int r) {

  std::vector<unsigned int> v = return_feature_indexes_for_grad_item_at_tkzjr_main(words, t, k, z);
  phi.insert(phi.end(), v.begin(), v.end());
  phi.push_back(return_feature_index_for_grad_item_at_tkzjr_tag_trainsition(z, r));

}


double semiCRF::calc_inner_product_at_tkzjr(const std::vector<std::string> &words,
                                            const int t, const int k, const int z,
                                            const int j, const int r) {
  double sum = 0.0;
  std::vector<unsigned int> v = return_feature_indexes_for_grad_item_at_tkzjr_main(words, t, k, z);
  for (auto itr = v.begin(); itr != v.end(); ++itr) {
    sum += weight[*itr];
  }

  sum += weight[return_feature_index_for_grad_item_at_tkzjr_tag_trainsition(z, r)];

  return sum;
}

void semiCRF::add_feature_index_for_grad_item_at_tkzjr(const std::vector<std::string> &words, const int t, const int k, const int z, const int j, const int r)
{

  std::vector<unsigned int> grad_item_item = return_feature_indexes_for_grad_item_at_tkzjr_main(words, t, k, z);
  grad_item_item.push_back(return_feature_index_for_grad_item_at_tkzjr_tag_trainsition(z, r));

  grad_item.push_back(grad_item_item);

}


double semiCRF::logsumexp(const std::vector<double> &v) {
  auto max_itr = std::max_element(v.begin(), v.end());
  double result = 0.0;
  for (auto itr = v.begin(); itr != v.end(); ++itr) {
    result += exp(*itr - *max_itr);
  }
  result = *max_itr + log(result);
  return result;
}


void semiCRF::calc_forward_backward(const std::vector<std::string> &words) {
  grad_item.clear();

  int alpha_size = words.size() * (MAXIMUM_LENGTH+1) * tag_index_size;
  int alpha_item_size = words.size() * (MAXIMUM_LENGTH+1) * (MAXIMUM_LENGTH+1) * tag_index_size * tag_index_size;
  //alphaは１次元のvectorを用いて3次元として表す alpha[t][k][z] = alpha[t*a1 + k*a2 + z]
  //alpha_itemは１次元のvectorを用いて5次元として表す alpha[t][k][z][j][r] = alpha_item[t*ai1 + k*ai2 + z*a3 + j*ai4 +r]
  alpha.resize(alpha_size, 0.0);
  beta.resize(alpha_size, 0.0);
  alpha_item.resize(alpha_item_size, 0.0);

  std::vector<double> lse_item_alpha;
  std::vector<double> lse_item_beta;

  for (int t = 0; t < words.size(); t++) {
    for (int k = 1; k < MAXIMUM_LENGTH+1; k++){
      if (t - k + 1 >= 1) {
        for (int z = 0; z < tag_index_size; z++) {
          for (int j = 0; j < MAXIMUM_LENGTH+1; j++) {
            if (t - k - j + 1>= 0) {
              for (int r = 0; r < tag_index_size; r++) {
                  //std::cout << t << " " << k << " " << z << " " << j << " " << r << " " << calc_inner_product_at_tkzjr(words, t, k, z, j, r) << std::endl;
                alpha_item[t*ai1 + k*ai2 + z*ai3 + j*ai4 + r] = calc_inner_product_at_tkzjr(words, t, k, z, j, r);
              }
            }
          }
        }
      }
    }
  }

  //以下alpha beta の計算
  //<BOS>からの遷移
  for (int t = 1; t < words.size()-1; t++) {
    int t_beta = words.size() -1 -t;
    for (int k = 1; k < MAXIMUM_LENGTH+1; k++){
      if (t - k + 1 == 1) {
        for (int z = 1; z < tag_index_size; z++) {
          int j = 1;
          int r = 0;
          add_feature_index_for_grad_item_at_tkzjr(words, t, k, z, j, r);
          alpha[t*a1 + k*a2 + z] = alpha_item[t*ai1 + k*ai2 + z*ai3 + j*ai4 + r];
          beta[t_beta*a1 + k*a2 + z] = alpha_item[(t_beta+j+k-1)*ai1 + j*ai2 + r*ai3 + k*ai4 + z];
        }
      }
    }
  }

  //alpha beta の計算
  for (int t = 1; t < words.size()-1; t++) {
    int t_beta = words.size() -1 -t;
    for (int k = 1; k < MAXIMUM_LENGTH+1; k++){
      if (t - k + 1 >= 2) {
        for (int z = 1; z < tag_index_size; z++) {
          lse_item_alpha.clear();
          lse_item_beta.clear();
          for (int j = 1; j < MAXIMUM_LENGTH+1; j++) {
            if (t - k - j + 1 >= 1) {
              for (int r = 1; r < tag_index_size; r++) {
                add_feature_index_for_grad_item_at_tkzjr(words, t, k, z, j, r);
                lse_item_alpha.push_back(alpha_item[t*ai1 + k*ai2 + z*ai3 + j*ai4 + r] + alpha[(t-k)*a1 + j*a2 + r]);
                lse_item_beta.push_back(alpha_item[(t_beta+j+k-1)*ai1 + j*ai2 + r*ai3 + k*ai4 + z] + beta[(t_beta+k)*a1 + j*a2 + r]);
              }
            }
          }
          alpha[t*a1 + k*a2 + z] = logsumexp(lse_item_alpha);
          beta[t_beta*a1 + k*a2 + z] = logsumexp(lse_item_beta);
        }
      }
    }
  }

  //<EOS>への遷移
  int t = words.size() -1;
  int t_beta = words.size() -1 -t;
  int k = 1;
  int z = 0;
  lse_item_alpha.clear();
  lse_item_beta.clear();
  for (int j = 1; j < MAXIMUM_LENGTH+1; j++) {
    if (t - k - j + 1 >= 1) {
      for (int r = 1; r < tag_index_size; r++) {
        add_feature_index_for_grad_item_at_tkzjr(words, t, k, z, j, r);
        lse_item_alpha.push_back(alpha_item[t*ai1 + k*ai2 + z*ai3 + j*ai4 + r] + alpha[(t-k)*a1 + j*a2 + r]);
        lse_item_beta.push_back(alpha_item[(t_beta+j+k-1)*ai1 + j*ai2 + r*ai3 + k*ai4 + z] + beta[(t_beta+k)*a1 + j*a2 + r]);
      }
    }
  }
  alpha[t*a1 + k*a2 + z] = logsumexp(lse_item_alpha);
  beta[t_beta*a1 + k*a2 + z] = logsumexp(lse_item_beta);
  assert(std::abs(alpha[t*a1 + k*a2 + z] - beta[t_beta*a1 + k*a2 + z]) < 0.001);
}


void semiCRF::update_weight_minus_grad(const std::vector<std::string> &words) {
  const double norm_z = beta[0 + 1*a2 + 0];
  unsigned int grad_item_index = 0;

  //BOSの計算
  for (int t = 1; t < words.size()-1; t++) {
    for (int k = 1; k < MAXIMUM_LENGTH+1; k++){
      if (t - k + 1 == 1) {
        for (int z = 1; z < tag_index_size; z++) {
          int j = 1;
          int r = 0;
          double grad_w = exp(alpha_item[t*ai1 + k*ai2 + z*ai3 + j*ai4 + r] + 0.0 + beta[(t-k+1)*a1 + k*a2 + z] - norm_z);
          assert(0.0 <= grad_w && grad_w <= 1.01);
          std::vector<unsigned int> &grad_item_item = grad_item[grad_item_index++];
          for (auto itr = grad_item_item.begin(); itr != grad_item_item.end(); ++itr) {
            weight[*itr] += ETA * (-1.0 * grad_w);
          }
        }
      }
    }
  }

  //alpha beta の計算
  for (int t = 1; t < words.size()-1; t++) {
    for (int k = 1; k < MAXIMUM_LENGTH+1; k++){
      if (t - k + 1 >= 2) {
        for (int z = 1; z < tag_index_size; z++) {
          for (int j = 1; j < MAXIMUM_LENGTH+1; j++) {
            if (t - k - j + 1 >= 1) {
              for (int r = 1; r < tag_index_size; r++) {
                double grad_w = exp(alpha_item[t*ai1 + k*ai2 + z*ai3 + j*ai4 + r] + alpha[(t-k)*a1 + j*a2 + r] + beta[(t-k+1)*a1 + k*a2 + z] - norm_z);
                assert(0.0 <= grad_w && grad_w <= 1.01);
                std::vector<unsigned int> &grad_item_item = grad_item[grad_item_index++];
                for (auto itr = grad_item_item.begin(); itr != grad_item_item.end(); ++itr) {
                  weight[*itr] += ETA * (-1.0 * grad_w);
                }
              }
            }
          }
        }
      }
    }
  }

  //<EOS>への遷移
  int t = words.size() -1;
  int k = 1;
  int z = 0;
  for (int j = 1; j < MAXIMUM_LENGTH+1; j++) {
    if (t - k - j + 1 >= 1) {
      for (int r = 1; r < tag_index_size; r++) {
        double grad_w = exp(alpha_item[t*ai1 + k*ai2 + z*ai3 + j*ai4 + r] + alpha[(t-k)*a1 + j*a2 + r] + 0.0 - norm_z);
        assert(0.0 <= grad_w && grad_w <= 1.01);
        std::vector<unsigned int> &grad_item_item = grad_item[grad_item_index++];
        for (auto itr = grad_item_item.begin(); itr != grad_item_item.end(); ++itr) {
          weight[*itr] += ETA * (-1.0 * grad_w);
        }
      }
    }
  }
  //assert(grad_item[grad_item_index] == *grad_item.end());
}


void semiCRF::update_weight_plus_grad(const std::vector<std::string> &words, const std::vector<labels> &tag_begin_ends) {
  phi.clear();
  for (int i = 1; i < tag_begin_ends.size(); i++) {
    create_feature_vec_at_tkzjr(words, tag_begin_ends[i].second.second, tag_begin_ends[i].second.second - tag_begin_ends[i].second.first + 1,
                                tag_begin_ends[i].first, tag_begin_ends[i-1].second.second, tag_begin_ends[i-1].first);
  }
  for(auto itr = phi.begin(); itr != phi.end(); ++itr) {
    weight[*itr] += ETA * 1.0;
  }
}


void semiCRF::update_l2norm() {
  for (auto itr = weight.begin(); itr != weight.end(); ++itr) {
    *itr -= ETA * (*itr * C);
  }
}

void semiCRF::train(const std::vector<std::string> &words, const std::vector<labels> &tag_begin_ends)
{
  /*
  for (int i = 1; i < words.size()-1;i++) {
    std::cout << words[i] << " ";
  }
  std::cout << '\n';
  */
  calc_forward_backward(words);
  update_weight_minus_grad(words);
  update_weight_plus_grad(words, tag_begin_ends);
  update_l2norm();
}

void semiCRF::viterbi(const std::vector<std::string> &words, std::vector<labels> &tag_begin_ends) {
  //forward algorithm
    int alpha_size = words.size() * (MAXIMUM_LENGTH+1) * tag_index_size;
  //alphaは１次元のvectorを用いて2次元として表す alpha[l][t] = alpha[l*words.size()+t]
  alpha.resize(alpha_size, 0.0);
  std::vector<int> alpha_back_pointer_at_t(alpha_size, -1);
  std::vector<int> alpha_back_pointer_at_k(alpha_size, -1);
  std::vector<int> alpha_back_pointer_at_z(alpha_size, -1);
  std::vector<double> lse_item_alpha;

  //以下alpha の計算
  //<BOS>からの遷移
  for (int t = 1; t < words.size()-1; t++) {
    for (int k = 1; k < MAXIMUM_LENGTH+1; k++){
      if (t - k + 1 == 1) {
        for (int z = 1; z < tag_index_size; z++) {
          int j = 1;
          int r = 0;
          alpha[t*a1 + k*a2 + z] = calc_inner_product_at_tkzjr(words, t, k, z, j, r) + 0.0;
          alpha_back_pointer_at_t[t*a1 + k*a2 + z] = t - k;
          alpha_back_pointer_at_k[t*a1 + k*a2 + z] = j;
          alpha_back_pointer_at_z[t*a1 + k*a2 + z] = r;
        }
      }
    }
  }

  //alpha の計算
  for (int t = 1; t < words.size()-1; t++) {
    for (int k = 1; k < MAXIMUM_LENGTH+1; k++){
      if (t - k + 1 >= 2) {
        for (int z = 1; z < tag_index_size; z++) {
          double max = -DBL_MAX;
          for (int j = 1; j < MAXIMUM_LENGTH+1; j++) {
            if (t - k - j + 1 >= 1) {
              for (int r = 1; r < tag_index_size; r++) {
                double score_at_tkzjr = calc_inner_product_at_tkzjr(words, t, k, z, j, r) + alpha[(t-k)*a1 + j*a2 + r];
                if (max < score_at_tkzjr) {
                  max = score_at_tkzjr;
                  alpha_back_pointer_at_t[t*a1 + k*a2 + z] = t - k;
                  alpha_back_pointer_at_k[t*a1 + k*a2 + z] = j;
                  alpha_back_pointer_at_z[t*a1 + k*a2 + z] = r;
                }
              }
            }
          }
          alpha[t*a1 + k*a2 + z] = max;
        }
      }
    }
  }

  //<EOS>への遷移
  int t = words.size() -1;
  int k = 1;
  int z = 0;
  double max = -DBL_MAX;
  for (int j = 1; j < MAXIMUM_LENGTH+1; j++) {
    if (t - k - j + 1 >= 1) {
      for (int r = 1; r < tag_index_size; r++) {
        double score_at_tkzjr = calc_inner_product_at_tkzjr(words, t, k, z, j, r) + alpha[(t-k)*a1 + j*a2 + r];
        if (max < score_at_tkzjr) {
          max = score_at_tkzjr;
          alpha_back_pointer_at_t[t*a1 + k*a2 + z] = t - k;
          alpha_back_pointer_at_k[t*a1 + k*a2 + z] = j;
          alpha_back_pointer_at_z[t*a1 + k*a2 + z] = r;
        }
      }
    }
  }
  alpha[t*a1 + k*a2 + z] = max;


  //backward algorithm
  t = words.size() -1;
  k = 1;
  z = 0;
  labels lbl(z, std::make_pair(t-k+1, t));
  while (t >= 0) {
    int t_item = alpha_back_pointer_at_t[t*a1 + k*a2 + z];
    int k_item = alpha_back_pointer_at_k[t*a1 + k*a2 + z];
    int z_item = alpha_back_pointer_at_z[t*a1 + k*a2 + z];
    t = t_item;
    k = k_item;
    z = z_item;
    lbl.first = z;
    lbl.second = std::make_pair(t-k+1, t);
    tag_begin_ends.push_back(lbl);
  }
  std::reverse(tag_begin_ends.begin(), tag_begin_ends.end());
}

void semiCRF::test(const std::vector<std::string> &words, std::vector<labels> &tag_begin_ends) {
  tag_begin_ends.clear();
  viterbi(words, tag_begin_ends);
}

std::vector<labels> semiCRF::crate_tag_begin_ends_from_partial_annotation(const std::vector<std::string> &words, const std::vector<labels> &partial_annotaion) {
  end2tag_begin.clear();
  begin2tag_end.clear();

  std::vector<labels> tag_begin_ends;
  labels lbl(0, std::make_pair(0, 0));

  //partial anntotaion されていない単語列に対して可能なラベル列を列挙
  int t = 0;
  for (int i = 0; i < partial_annotaion.size(); i++) {
    int next_z = partial_annotaion[i].first;
    int next_begin = partial_annotaion[i].second.first;
    int next_end = partial_annotaion[i].second.second;
    for (int begin = t; begin < next_begin; begin++) {
      for (int end = begin; end < std::min((begin+MAXIMUM_LENGTH), next_begin); end++) {
        for (int z = 1; z < tag_index_size; z++) {
          lbl.first = z;
          lbl.second = std::make_pair(begin, end);
          tag_begin_ends.push_back(lbl);

          end2tag_begin[end].push_back(std::make_pair(z, begin));
          begin2tag_end[begin].push_back(std::make_pair(z, end));
        }
      }
    }
    tag_begin_ends.push_back(partial_annotaion[i]);
    end2tag_begin[next_end].push_back(std::make_pair(next_z, next_begin));
    begin2tag_end[next_begin].push_back(std::make_pair(next_z, next_end));
    t = next_end + 1;
  }

  return tag_begin_ends;
}

void semiCRF::calc_forward_backward_with_patial_annotaion(const std::vector<std::string> &words, const std::vector<labels> &tag_begin_ends) {
  int alpha_size = words.size() * (MAXIMUM_LENGTH+1) * tag_index_size;
  //alphaは１次元のvectorを用いて3次元として表す alpha[t][k][z] = alpha[t*a1 + k*a2 + z]
  alpha_partial.resize(alpha_size, 0.0);
  beta_partial.resize(alpha_size, 0.0);

  std::vector<double> lse_item_alpha;
  std::vector<double> lse_item_beta;

  //alpha_partial and beta_partial の計算
  for (int i = 1; i < tag_begin_ends.size(); i++) {
    int i_beta = tag_begin_ends.size() - i -1;
    int t = tag_begin_ends[i].second.second;
    int k = -1 * (tag_begin_ends[i].second.first -t -1);
    int z = tag_begin_ends[i].first;

    int t_beta = tag_begin_ends[i_beta].second.first;
    int k_beta = tag_begin_ends[i_beta].second.second - t_beta + 1;
    int z_beta = tag_begin_ends[i_beta].first;
    if (t-k == -1) {
      continue;
    }

    lse_item_alpha.clear();
    lse_item_beta.clear();

    for (auto itr = end2tag_begin[t-k].begin(); itr != end2tag_begin[t-k].end(); ++itr) {
      int r = itr->first;
      int j = -1 * (itr->second -t +k -1);
      lse_item_alpha.push_back(alpha_item[t*ai1 + k*ai2 + z*ai3 + j*ai4 + r] + alpha_partial[(t-k)*a1 + j*a2 + r]);
    }
    for (auto itr = begin2tag_end[t_beta+k_beta].begin(); itr != begin2tag_end[t_beta+k_beta].end(); ++itr) {
      int r_beta = itr->first;
      int j_beta = itr->second -t_beta -k_beta +1;
      lse_item_beta.push_back(alpha_item[(t_beta+j_beta+k_beta-1)*ai1 + j_beta*ai2 + r_beta*ai3 + k_beta*ai4 + z_beta] + beta_partial[(t_beta+k_beta)*a1 + j_beta*a2 + r_beta]);
    }

    alpha_partial[t*a1 + k*a2 + z] = logsumexp(lse_item_alpha);
    beta_partial[t_beta*a1 + k_beta*a2 + z_beta] = logsumexp(lse_item_beta);
  }

  assert(std::abs(alpha_partial[(words.size() -1)*a1 + 1*a2 + 0] - beta_partial[0*a1 + 1*a2 + 0]) < 0.001);
}


void semiCRF::update_weight_plus_grad_with_partial_annotation(const std::vector<std::string> &words, const std::vector<labels> &tag_begin_ends) {
  const double norm_z = beta_partial[0 + 1*a2 + 0];
  for (auto lbl = tag_begin_ends.begin(); lbl != tag_begin_ends.end(); ++lbl) {
    int t = lbl->second.second;
    int k = -1 * (lbl->second.first -t -1);
    int z = lbl->first;

    //feature に対する重みの main 部分の更新
    const double gamma = exp(alpha_partial[t*a1 + k*a2 + z] + beta_partial[(t-k+1)*a1 + k*a2 + z] - norm_z);
    assert(0.0 <= gamma && gamma <= 1.01);
    std::vector<unsigned int> v = return_feature_indexes_for_grad_item_at_tkzjr_main(words, t, k, z);
    for(auto itr = v.begin(); itr != v.end(); ++itr) {
      weight[*itr] += ETA * gamma;
    }

    //tag trainsion 部分の重みの更新
    for (auto itr = end2tag_begin[t-k].begin(); itr != end2tag_begin[t-k].end(); ++itr) {
      int r = itr->first;
      int j = -1 * (itr->second -t +k -1);
      const double epsilon = exp(alpha_item[t*ai1 + k*ai2 + z*ai3 + j*ai4 + r] + alpha_partial[(t-k)*a1 + j*a2 + r] + beta_partial[(t-k+1)*a1 + k*a2 + z] - norm_z);
      assert(0.0 <= epsilon && epsilon <= 1.01);
      weight[return_feature_index_for_grad_item_at_tkzjr_tag_trainsition(z, r)] += ETA * epsilon;
    }
  }
}


void semiCRF::partial_train(const std::vector<std::string> &words, const std::vector<labels> &partial_annotaion) {
  std::vector<labels> tag_begin_ends = crate_tag_begin_ends_from_partial_annotation(words, partial_annotaion);
  calc_forward_backward(words);
  calc_forward_backward_with_patial_annotaion(words, tag_begin_ends);
  update_weight_minus_grad(words);
  update_weight_plus_grad_with_partial_annotation(words, tag_begin_ends);
  //update_l2norm();
}

void semiCRF::serialize(FILE *fp) {
  if (!fp) {
    std::cout << "file not exist" << '\n';
    assert(false);
  }
  if (fwrite(&word_index_size, sizeof(unsigned int), 1, fp) != 1) {
    std::cout << "failed to write word_index_size" << '\n';
    assert(false);
  }
  if (fwrite(&tag_index_size, sizeof(int), 1, fp) != 1) {
    std::cout << "failed to write tag_index_size" << '\n';
    assert(false);
  }
  for (int i = 0; i < weight.size(); i++) {
    if (fwrite(&weight[i], sizeof(double), 1, fp) != 1) {
      std::cout << "failed to write weight" << '\n';
      assert(false);
    }
  }
  for (auto itr = w2i.begin(); itr != w2i.end(); ++itr) {
    int w_len = itr->first.size();
    if (fwrite(&w_len, sizeof(int), 1, fp) != 1) {
      std::cout << "failed to write w2i 1" << '\n';
      assert(false);
    }
    const char *save_word = itr->first.c_str();
    for (int j = 0; j < w_len; j++) {
      if (fputc(save_word[j], fp) < 0) {
        std::cout << "failed to write w2i 2" << '\n';
        assert(false);
      }
    }
    if (fwrite(&(itr->second), sizeof(unsigned int), 1, fp) != 1) {
      std::cout << "failed to write w2i 3" << '\n';
      assert(false);
    }
  }

}

void semiCRF::unserialize(FILE *fp) {
  if (!fp) {
    std::cout << "file not exist" << '\n';
    assert(false);
  }
  if (fread(&word_index_size, sizeof(unsigned int), 1, fp) != 1) {
    std::cout << "failed to read word_index_size" << '\n';
    assert(false);
  }
  if (fread(&tag_index_size, sizeof(int), 1, fp) != 1) {
    std::cout << "failed to read tag_index_size" << '\n';
    assert(false);
  }
  feature_size_onetag = (6 * word_index_size) + tag_index_size;   //chunk_b^e, w_b, ..., w_e, w_b-1, w_e+1, tag_t-1
  feature_size = feature_size_onetag * tag_index_size;
  weight.resize(feature_size, 0.0);

  a2 = tag_index_size;
  a1 = MAXIMUM_LENGTH * a2;

  ai4 = tag_index_size;
  ai3 = MAXIMUM_LENGTH * ai4;
  ai2 = tag_index_size * ai3;
  ai1 = MAXIMUM_LENGTH * ai2;
  for (int i = 0; i < weight.size(); i++) {
    if (fread(&weight[i], sizeof(double), 1, fp) != 1) {
      std::cout << "failed to read weight" << '\n';
      assert(false);
    }
  }
  for (unsigned int i = 0; i < word_index_size; i++) {
    int w_len = -1;
    if (fread(&w_len, sizeof(int), 1, fp) != 1) {
      std::cout << "failed to read w2i 1" << '\n';
      assert(false);
    }
    char load_word[w_len+1];
    for (int j = 0; j < w_len; j++) {
      if ((load_word[j] = fgetc(fp)) == EOF) {
        std::cout << "failed to read w2i 2" << '\n';
        assert(false);
      }
    }
    load_word[w_len] = '\0';
    unsigned int load_word_index;
    if (fread(&load_word_index, sizeof(unsigned int), 1, fp) != 1) {
      std::cout << "failed to read w2i 3" << '\n';
      assert(false);
    }
    w2i[std::string(load_word, w_len)] = load_word_index;
  }
}

std::vector<double> semiCRF::calc_semicrf_score(const std::vector<std::string> &words) {
  int alpha_item_size = words.size() * (MAXIMUM_LENGTH+1) * (MAXIMUM_LENGTH+1) * tag_index_size * tag_index_size;
  //alphaは１次元のvectorを用いて3次元として表す alpha[t][k][z] = alpha[t*a1 + k*a2 + z]
  //alpha_itemは１次元のvectorを用いて5次元として表す alpha[t][k][z][j][r] = alpha_item[t*ai1 + k*ai2 + z*a3 + j*ai4 +r]
  alpha_item.resize(alpha_item_size, 0.0);

  for (int t = 0; t < words.size(); t++) {
    for (int k = 1; k < MAXIMUM_LENGTH+1; k++){
      if (t - k + 1 >= 1) {
        for (int z = 0; z < tag_index_size; z++) {
          for (int j = 0; j < MAXIMUM_LENGTH+1; j++) {
            if (t - k - j + 1>= 0) {
              for (int r = 0; r < tag_index_size; r++) {
                alpha_item[t*ai1 + k*ai2 + z*ai3 + j*ai4 + r] = calc_inner_product_at_tkzjr(words, t, k, z, j, r);
              }
            }
          }
        }
      }
    }
  }

  return alpha_item;
}

int semiCRF::get_ai1() {return ai1;}
int semiCRF::get_ai2() {return ai2;}
int semiCRF::get_ai3() {return ai3;}
int semiCRF::get_ai4() {return ai4;}
