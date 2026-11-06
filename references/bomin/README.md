***기본모델***
"bleu_ko_mecab": 22.1501,
"chrf": 30.7359
***fine-tuning 1520 step후의 모델***
"bleu_ko_mecab": 26.8188,
"chrf": 35.1142

- 1520이후에는 eval-valid loss값이 상승하고 test 결과가 감소하므로 1520에서 stop 하였음
- 이후의 finetuning에서 chrf를 50이상으로 만들 수 있으면 좋을거 같음
- train및 test방식은 코드 참조바람
