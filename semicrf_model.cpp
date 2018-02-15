#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <assert.h>
#include <unistd.h>
#include "semicrf.hpp"
#include "../../preprocess/sentence.hpp"

using labels = std::pair<int, std::pair<int, int>>;

int main(int argc, char *argv[]) {
  int opt;
  std::string train_file_name;
  std::string partial_train_file_name;
  std::string test_file_name;
  std::string save_model_file_name;
  while((opt = getopt(argc, argv, "l:p:t:m:h")) != -1) {
    switch(opt) {
      case 'l':
        train_file_name = optarg;
        std::cout << "学習ファイル名 = " << train_file_name << '\n';
        break;
      case 'p':
        partial_train_file_name = optarg;
        std::cout << "学習ファイル名(部分アノテーション) = " << partial_train_file_name << '\n';
        break;
      case 't':
        test_file_name = optarg;
        std::cout << "テストファイル名 = " << test_file_name << '\n';
        break;
      case 'm':
        save_model_file_name = optarg;
        std::cout << "モデルファイル名 = " << save_model_file_name << '\n';
        break;
      case 'h':
        std::cout << "help" << '\n';
        std::cout << "-l = 学習ファイル名 (例: 京都-府/Loc 京都-市/Loc どちら/O の/O 京都/Loc です/O か/O)" << '\n';
        std::cout << "-p = 学習ファイル名(部分アノテーション) (例: どちら でも なく 京都-大学/Org でし た)" << '\n';
        std::cout << "-t = テストファイル名" << '\n';
        std::cout << "-h = help" << '\n';
        return -1;
        break;
    }
  }

  std::vector<std::vector<std::string>> dataset;
  std::vector<std::vector<unsigned int>> dataset_index;
  std::vector<std::vector<labels>> dataset_tag_index;

  std::vector<std::vector<std::string>> dataset_p;
  std::vector<std::vector<unsigned int>> dataset_p_index;
  std::vector<std::vector<labels>> dataset_p_tag_index;

  std::vector<std::vector<unsigned int>> dataset_test_index;

  std::map<std::string, unsigned int> word2index;
  std::map<unsigned int, std::string> index2word;
  std::map<std::string, int> tag2index;
  std::map<int, std::string> index2tag;
  unsigned int word_index_size = 0;
  int tag_index_size = 0;

  word2index["<BOS>"] = word_index_size;
  index2word[word_index_size] = "<BOS>";
  word_index_size++;
  tag2index["<BOS>"] = tag_index_size;
  index2tag[tag_index_size] = "<BOS>";
  tag_index_size++;

  word2index["<UNK>"] = word_index_size;
  index2word[word_index_size] = "<UNK>";
  word_index_size++;

  std::ifstream train_file(train_file_name);
  std::string str;
  if (train_file_name != "") {
    if (train_file.fail()){
      std::cerr << "学習ファイル読み込み失敗" << std::endl;
      return -1;
    }
  }

  std::ifstream partial_train_file(partial_train_file_name);
  if (partial_train_file_name != "") {
    if (partial_train_file.fail()){
      std::cerr << "学習ファイル(部分アノテーション)読み込み失敗" << std::endl;
      return -1;
    }
  }

  std::ifstream test_file(test_file_name);
  if (test_file_name != "") {
    if (test_file.fail()){
      std::cerr << "テストファイル読み込み失敗" << std::endl;
      return -1;
    }
  }

  //学習ファイル読み込み
  if (train_file_name != "") {
  while (getline(train_file, str)) {
    std::vector<std::string> tokens = preprocess::split_sentence(str, " ");

    std::vector<std::string> words;
    std::vector<unsigned int> words_index;
    std::vector<labels> tags_index;
    int t = 0;

    words.push_back("<BOS>");
    words_index.push_back(word2index["<BOS>"]);
    labels lbl(tag2index["<BOS>"], std::make_pair(t, t));
    tags_index.push_back(lbl);
    t++;
    for(auto token_itr = tokens.begin(); token_itr != tokens.end(); ++token_itr) {
      std::vector<std::string> chunk_and_tag = preprocess::split_sentence(*token_itr, "/");
      assert(chunk_and_tag.size() == 2);

      std::string chunk = preprocess::replace_sentence(chunk_and_tag[0], "-", "");
      auto itr1 = word2index.find(chunk);
      if (itr1 == word2index.end()) {
        word2index[chunk] = word_index_size;
        index2word[word_index_size] = chunk;
        word_index_size++;
      }

      auto itr2 = tag2index.find(chunk_and_tag[1]);
      if (itr2 == tag2index.end()) {
        tag2index[chunk_and_tag[1]] = tag_index_size;
        index2tag[tag_index_size] = chunk_and_tag[1];
        tag_index_size++;
      }

      std::vector<std::string> chunk_sp = preprocess::split_sentence(chunk_and_tag[0], "-");
      for (auto chunk_sp_itr = chunk_sp.begin(); chunk_sp_itr != chunk_sp.end(); ++chunk_sp_itr) {
        auto itr3 = word2index.find(*chunk_sp_itr);
        if (itr3 == word2index.end()) {
          word2index[*chunk_sp_itr] = word_index_size;
          index2word[word_index_size] = *chunk_sp_itr;
          word_index_size++;
        }
        words.push_back(*chunk_sp_itr);
        words_index.push_back(word2index[*chunk_sp_itr]);
      }
      lbl.first = tag2index[chunk_and_tag[1]];
      lbl.second = std::make_pair(t, t+chunk_sp.size()-1);
      //tags.push_back(word_and_tag[1]);
      tags_index.push_back(lbl);
      t = t + chunk_sp.size();
    }


    words.push_back("<BOS>");
    words_index.push_back(word2index["<BOS>"]);
    lbl.first = tag2index["<BOS>"];
    lbl.second = std::make_pair(t, t);
    tags_index.push_back(lbl);
    dataset.push_back(words);
    dataset_index.push_back(words_index);
    dataset_tag_index.push_back(tags_index);
  }
  }

  if (partial_train_file_name != "") {
    //学習ファイル読み込み
    while (getline(partial_train_file, str)) {
      std::vector<std::string> tokens = preprocess::split_sentence(str, " ");

      std::vector<std::string> words;
      std::vector<unsigned int> words_index;
      std::vector<labels> tags_index;
      int t = 0;

      words.push_back("<BOS>");
      words_index.push_back(word2index["<BOS>"]);
      labels lbl(tag2index["<BOS>"], std::make_pair(t, t));
      tags_index.push_back(lbl);
      t++;
      for(auto token_itr = tokens.begin(); token_itr != tokens.end(); ++token_itr) {
        std::vector<std::string> chunk_and_tag = preprocess::split_sentence(*token_itr, "/");
        assert(chunk_and_tag.size() == 2 or chunk_and_tag.size() == 1);
        if (chunk_and_tag.size() == 2) {
          std::string chunk = preprocess::replace_sentence(chunk_and_tag[0], "-", "");
          auto itr1 = word2index.find(chunk);
          if (itr1 == word2index.end()) {
            word2index[chunk] = word_index_size;
            index2word[word_index_size] = chunk;
            word_index_size++;
          }
          auto itr2 = tag2index.find(chunk_and_tag[1]);
          if (itr2 == tag2index.end()) {
            tag2index[chunk_and_tag[1]] = tag_index_size;
            index2tag[tag_index_size] = chunk_and_tag[1];
            tag_index_size++;
          }

          std::vector<std::string> chunk_sp = preprocess::split_sentence(chunk_and_tag[0], "-");
          for (auto chunk_sp_itr = chunk_sp.begin(); chunk_sp_itr != chunk_sp.end(); ++ chunk_sp_itr) {
            auto itr3 = word2index.find(*chunk_sp_itr);
            if (itr3 == word2index.end()) {
              word2index[*chunk_sp_itr] = word_index_size;
              index2word[word_index_size] = *chunk_sp_itr;
              word_index_size++;
            }
            words.push_back(*chunk_sp_itr);
            words_index.push_back(word2index[*chunk_sp_itr]);
          }

          std::vector<std::string> tag_sp = preprocess::split_sentence(chunk_and_tag[1], "|");
          for (auto tag_sp_itr = tag_sp.begin(); tag_sp_itr != tag_sp.end(); ++tag_sp_itr ) {
            auto itr2 = tag2index.find(*tag_sp_itr);
            if (itr2 == tag2index.end()) {
              tag2index[*tag_sp_itr] = tag_index_size;
              index2tag[tag_index_size] = *tag_sp_itr;
              tag_index_size++;
            }
            lbl.first = tag2index[*tag_sp_itr];
            lbl.second = std::make_pair(t, t+chunk_sp.size()-1);
            //tags.push_back(word_and_tag[1]);
            tags_index.push_back(lbl);
          }

          t = t + chunk_sp.size();
        }
        else if (chunk_and_tag.size() == 1) {
          auto itr1 = word2index.find(chunk_and_tag[0]);
          if (itr1 == word2index.end()) {
            word2index[chunk_and_tag[0]] = word_index_size;
            index2word[word_index_size] = chunk_and_tag[0];
            word_index_size++;
          }
          words.push_back(chunk_and_tag[0]);
          words_index.push_back(word2index[chunk_and_tag[0]]);
          t = t + 1;
        }
      }

      words.push_back("<BOS>");
      words_index.push_back(word2index["<BOS>"]);
      lbl.first = tag2index["<BOS>"];
      lbl.second = std::make_pair(t, t);
      tags_index.push_back(lbl);
      dataset_p.push_back(words);
      dataset_p_index.push_back(words_index);
      dataset_p_tag_index.push_back(tags_index);
    }
  }

  /*
  //テストファイル読み込み
  while (getline(test_file, str)) {
    std::vector<std::string> tokens = preprocess::split_sentence(str, " ");

    std::vector<std::string> words;
    std::vector<std::string> tags;
    std::vector<unsigned int> words_index;
    std::vector<int> tags_index;

    words_index.push_back(word2index["<BOS>"]);
    tags_index.push_back(tag2index["<BOS>"]);
    for(auto token_itr = tokens.begin(); token_itr != tokens.end(); ++token_itr) {
      std::vector<std::string> chunk_and_tag = preprocess::split_sentence(*token_itr, "/");
      assert(chunk_and_tag.size() == 2);

      auto itr1 = word2index.find(chunk_and_tag[0]);
      auto itr2 = tag2index.find(chunk_and_tag[1]);
      if (itr2 == tag2index.end()) {
        std::cerr << "テストファイル読み込み失敗 2" << std::endl;
      }

      std::vector<std::string> chunk_sp = preprocess::split_sentence(chunk_and_tag[0], "-");
      for (auto chunk_sp_itr = chunk_sp.begin(); chunk_sp_itr != chunk_sp.end(); ++ chunk_sp_itr) {
        auto itr3 = word2index.find(*chunk_sp_itr);
        if (itr3 == word2index.end()) {
          *chunk_sp_itr = "<UNK>";
        }
        words.push_back(*chunk_sp_itr);
        words_index.push_back(word2index[*chunk_sp_itr]);
        tags.push_back(chunk_and_tag[1]);
        tags_index.push_back(tag2index[chunk_and_tag[1]]);
      }

      //tags.push_back(word_and_tag[1]);
      //tags_index.push_back(tag2index[word_and_tag[1]]);
    }
    //dataset.push_back(words);
    //dataset_tag.push_back(tags);

    words_index.push_back(word2index["<BOS>"]);
    //tags_index.push_back(tag2index["<BOS>"]);
    dataset_index_test.push_back(words_index);
    //dataset_tag_index.push_back(tags_index);
  }*/

  semiCRF model(word_index_size, tag_index_size, word2index);

  for (int i = 0; i < 20; i++) {
    if (train_file_name != "") {
        std::vector<int> rand_index(dataset.size());
        for (int j = 0; j < dataset.size(); j++) {
            rand_index[j] = j;
        }
        random_shuffle(rand_index.begin(), rand_index.end());

        for (int j = 0; j < dataset_index.size(); j++) {
            model.train(dataset[rand_index[j]], dataset_tag_index[rand_index[j]]);
        }
    }
    if (partial_train_file_name != "") {
        std::vector<int> rand_index(dataset_p.size());
        for (int j = 0; j < dataset_p.size(); j++) {
            rand_index[j] = j;
        }
        random_shuffle(rand_index.begin(), rand_index.end());
        for (int j = 0; j < dataset_p_index.size(); j++) {
            model.partial_train(dataset_p[rand_index[j]], dataset_p_tag_index[rand_index[j]]);
        }
    }
  }

  if (save_model_file_name != "") {
    FILE *mfp = NULL;
    if ((mfp = fopen(save_model_file_name.c_str(), "wb")) == NULL) {
        std::cout << "モデル保存エラー" << '\n';
        return -1;
    }
    model.serialize(mfp);
    fclose(mfp);

  }


  if (save_model_file_name != "") {
    FILE *mfp = NULL;
    if ((mfp = fopen(save_model_file_name.c_str(), "rb")) == NULL) {
        std::cout << "モデル読み込みエラー" << '\n';
        return -1;
    }
  semiCRF load_model(mfp);
  fclose(mfp);
  std::cout << "load done" << std::endl;


    std::vector<labels> predict_labels;
    for (int j = 0; j < dataset_p_index.size(); j++) {
      model.test(dataset_p[j], predict_labels);
      for(auto itr = predict_labels.begin()+2; itr != predict_labels.end(); ++itr) {
        for(int t = itr->second.first; t <= itr->second.second; t++){
          if (t < itr->second.second) {
              std::cout << dataset_p[j][t] << "-";
          } else {
              std::cout << dataset_p[j][t] << "/" << index2tag[itr->first] << " ";
          }
        }
        //std::cout << itr->first << "~" << itr->second.first << "~" << itr->second.second << "\n";
      }
      std::cout << '\n';
    }
  }

  return 0;
}
