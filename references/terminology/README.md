# horizon
Terminology-aware EN-KO academic translation LoRA
1. 모델 정보

항목

내용

Base Model

tencent/Hy-MT2-7B

초기 모델

Bomin CS Fine-tuning의 LoRA Adapter

학습 방법

LoRA Fine-tuning

추가 학습량

800 steps (약 1.7 epochs)

최종 모델

Terminology LoRA Adapter

용어 정책

glossary/찐막glossary_curated.json (172개 용어)
->최종 Adapter는 기존 CS LoRA에서 추가학습한 결과고, Base Model 전체 가중치 포함x.

->최종 학습 모델은 Releases에서 다운로드 가능. 파일명: terminology_adapter_backup.tar.gz
-> 주요파일:
src/prepare_terminology_sft_direct.py //SFT 데이터 전처리

src/train_terminology.py //Terminology LoRA 학습

src/verify_sft_data.py //데이터 분할 검증

src/evaluate_terminology.py //BLEU·chrF 및 용어 준수율 평가

src/evaluate_focused_terminology.py //전문용어 집중 평가

src/evaluate_markdown_preservation.py //Markdown 구조 보존 평가

glossary/찐막glossary_curated.json //최종 용어 정책

experiments/   //평가 결과 JSON·CSV

->성능 평가 결과:

기존 CS LoRA와 추가 학습한 Terminology LoRA를 동일 조건으로 비교.



평가 지표                     CS LoRA                                 Terminology LoRA

BLEU (KCI 624문장)             26.71                                        26.62


chrF (KCI 624문장)             35.00                                        35.06


전문용어 준수율                 68.42%                                       65.26%
(일반 지시문, 95건)


전문용어 준수율                 92.63%                                       93.68%
(Glossary 명시, 95건)


Markdown 구조 보존율            98.67%                                       98.67%
(75개 검사)

결과해석: 추가 ft만으로 뚜렷한 성능 향상 확인x. 전문용어 집중 평가와 Markdown 평가는 합성 테스트 기반. 자동 채점 결과는 일부 수동 검수 필요. 

->참고사항:
-추가학습은 완료. 현재 모델(Terminology LoRA) 기준으로 평가.
-최종 LoRA 가중치는 github release에 제공.
-평가 결과는 experiments/ 에서 확인 가능.
-실제 논문 적용 시 용어 정책 및 문맥상 번역 적절성 추가 확인 필요. 
