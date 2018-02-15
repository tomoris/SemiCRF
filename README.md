# SemiCRF

## 概要  
セミマルコフモデルのCRF  
部分アノテーションからも学習可能  

## コンパイル  
`g++ -std=c++11 -Wall semicrf_model.cpp semicrf.cpp`

## 使い方  
`./a.out -p sample_partial_train_semicrf.txt -l sample_train_semicrf.txt -m save.model`
