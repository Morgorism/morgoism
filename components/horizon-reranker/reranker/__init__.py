"""Document-level N-best reranker (Yu et al., TACL 2020 style) for the team's Hy-MT2-7B LoRA model.

Pipeline: segment -> generate (N-best per sentence) -> score (features) -> tune (weights on valid) -> decode.
Owner: 안지후.
"""
