# 업로드받은 ZIP 5개와 최종 파이프라인의 관계

1. `PDF_to_Markdown-main.zip`
   - `PDF_to_Markdown/make_paper_md.py` → `components/pdf_to_markdown/make_paper_md.py`
   - `Markdown_to_PDF/pdf.py` → `components/markdown_to_pdf/pdf.py`
   - 실제 sample PDF/MinerU ZIP → `examples/`
2. `finetuning_Bomin-main.zip`
   - GPU 실행에 **역번역 점수용 CS LoRA 가중치**가 필요함. ZIP의 `adapter_model.safetensors`는 LFS 포인터이므로 포함하지 않음.
   - 학습·평가 코드 일부 → `references/bomin/` (참고용, 파이프라인에서 다시 학습하지 않음)
3. `horizon-main.zip`
   - 최종 용어 정책 → `components/glossary_curated.json`
   - 학습 코드 일부 → `references/terminology/` (참고용)
   - Terminology 모델은 후속 Stage4 추가 학습의 기반이며, 별도 번역 모델 프로세스로 3중 호출하지 않음.
4. `horizon-context-main.zip`
   - Stage4 Replay 추론 프롬프트/모델 학습 방식 확인 → `references/context/` (참고용)
   - 실제 Stage4 Replay LoRA 가중치는 GitHub Releases에서 별도 준비해야 함.
5. `horizon-reranker-main.zip`
   - `reranker/`, `evaluation/`, `configs/final.json`, `results/final/weights.json`, `setup_runpod.sh` 등 → `components/horizon-reranker/`
   - Stage4 번역 후보 + Bomin 역번역 점수 + LM 문맥 + 용어집 + 길이 가중치를 포함하는 **E_all 최종 파이프라인** 호출

모든 원본 소스는 업로드받은 각 ZIP에서 복사했고, 통합 편의를 위한 새 코드는 최상위 `run_pipeline.py`와 `scripts/make_rerank_config.py`, `tests/test_pipeline.py`입니다.
