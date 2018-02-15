# Semi Markov CRF

## 概要  
セミマルコフモデルのCRF  
(複数の単語からなるチャンクに対し1つのラベルを付与できる)  
部分アノテーションからも学習可能  

## コンパイル  
`g++ -std=c++11 -Wall semicrf_model.cpp semicrf.cpp`  

## 使い方  
`./a.out -p sample_partial_train_semicrf.txt -l sample_train_semicrf.txt -m save.model`  

## アノテーション方法  
単語はスペース区切り、「/」(スラッシュ)の後にラベルをつける  
`スキー/w-sports が/O したい/O です/O 。/O`  
チャンクに対しラベルをアノテーションするときは、「-」(ハイフン)で繋ぐ  
`ビーチ バレー => ビーチ-バレー`  
`ビーチ-バレー/s-sports が/O したい/O です/O 。/O`  
  
部分的にアノテーションすることも可能  
`私 は サーフィン/s-sports に 挑戦 したい です 。`  
  
曖昧なものには複数のラベルを学習可能  
「|」(縦棒、パイプ)でラベルを列挙  
`サッカー/s-sports|w-sports`  
