#include <iostream>
#include <vector>
#include <math.h>
#include <assert.h>
#include <cfloat>
#include <map>
#include <algorithm>
//#include "../preprocess/sentence.hpp"

using labels = std::pair<int, std::pair<int, int>>;

class semiCRF
{
private:
  unsigned int word_index_size;
  int tag_index_size;
  std::map<std::string, unsigned int> w2i;

  unsigned int feature_size;
  unsigned int feature_size_onetag;

  std::vector<double> phi;      //素性ベクトル
  std::vector<double> weight;
  const double ETA = 0.1;  //学習率
  const double C = 0.001; //正則化係数
  const int MAXIMUM_LENGTH = 6;
    //const int MAXIMUM_LENGTH = 2;

  int a2 = 0;
  int a1 = 0;

  int ai4 = 0;
  int ai3 = 0;
  int ai2 = 0;
  int ai1 = 0;

  std::vector<double> alpha;
  std::vector<double> beta;
  std::vector<double> alpha_item;  //すべてのパスのexp(w*phi)を保存 alpha[l][m][t] (高速化のため１次元)

  std::vector<double> alpha_partial;
  std::vector<double> beta_partial;

  std::vector<std::vector<std::vector<double>>> alpha_vec;
  std::vector<std::vector<std::vector<double>>> beta_vec;
  std::vector<std::vector<std::vector<std::vector<double>>>> alpha_item_vec;

  std::vector<std::vector<unsigned int>> grad_item;  //起こりうるtag遷移すべての場合の素性インデックスを格納

  std::map<int, std::vector<std::pair<int, int>>> end2tag_begin;
  std::map<int, std::vector<std::pair<int, int>>> begin2tag_end;

  std::vector<unsigned int> return_feature_indexes_for_grad_item_at_tkzjr_main(const std::vector<std::string> &words, const int t, const int k, const int z);
  unsigned int return_feature_index_for_grad_item_at_tkzjr_tag_trainsition(const int z, const int r);

  void create_feature_vec_at_tkzjr(const std::vector<std::string> &words, const int t, const int k, const int z, const int j, const int r);
  double calc_inner_product_at_tkzjr(const std::vector<std::string> &words, const int t, const int k, const int j, const int z, const int r);
  void add_feature_index_for_grad_item_at_tkzjr(const std::vector<std::string> &words, const int t, const int k, const int z, const int j, const int r);
  double logsumexp(const std::vector<double> &v);

  void calc_forward_backward(const std::vector<std::string> &words);

  void update_l2norm();
  void update_weight_minus_grad(const std::vector<std::string> &words);
  void update_weight_plus_grad(const std::vector<std::string> &words, const std::vector<labels> &tag_begin_ends);

  void viterbi(const std::vector<std::string> &words, std::vector<labels> &tag_begin_ends);

  std::vector<labels> crate_tag_begin_ends_from_partial_annotation(const std::vector<std::string> &words, const std::vector<labels> &partial_annotaion);
  void calc_forward_backward_with_patial_annotaion(const std::vector<std::string> &words, const std::vector<labels> &tag_begin_ends);
  void update_weight_plus_grad_with_partial_annotation(const std::vector<std::string> &words, const std::vector<labels> &tag_begin_ends);
  void unserialize(FILE *fp);

public:
  semiCRF(const unsigned int w_i_s, const int t_i_s, const std::map<std::string, unsigned int> &word2index);
  semiCRF(FILE *fp);
  void train(const std::vector<std::string> &words, const std::vector<labels> &tag_begin_ends);
  void test(const std::vector<std::string> &words, std::vector<labels> &tag_begin_ends);
  void partial_train(const std::vector<std::string> &words, const std::vector<labels> &partial_annotaion);
  void serialize(FILE *fp);
  std::vector<double> calc_semicrf_score(const std::vector<std::string> &words);
  int get_ai1();
  int get_ai2();
  int get_ai3();
  int get_ai4();
};
