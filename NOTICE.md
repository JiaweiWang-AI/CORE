# Attribution

CORE: Wang et al., *Causal-driven Large Language Models with Faithful Reasoning for Knowledge Question Answering*, ACM MM 2024. [DOI 10.1145/3664647.3681263](https://doi.org/10.1145/3664647.3681263).

The README framework image, `assets/core_framework.png`, reproduces Figure 3 from page 4 of the published paper. It is rendered directly from the PDF without changing the figure content; its source and rendering coordinates are recorded in `PROVENANCE.json`.

ScienceQA: Lu et al., *Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering*, NeurIPS 2022. [Official repository](https://github.com/lupantech/ScienceQA). Inherited prompt-building conventions retain the original MIT notice in `licenses/ScienceQA-CODE-MIT.txt`. Downloaded datasets retain their own terms; the downloader preserves LICENSE-DATA. No dataset is bundled.

The causal-evaluation/refinement prompt constants and the ScienceQA builder originate from the authors' supplementary code. The orchestration and verified final-selection reconstruction were organized in 2026, with evidence and source hashes in PROVENANCE.json.

The paper uses InstructBLIP (Dai et al., NeurIPS 2023) to obtain visual descriptions. The downloader retrieves the verified public MM-CoT captions from `amazon-science/mm-cot` commit `8dd4ac02b94f21347973491f6e6b828502d23f9d`. Captions and model weights are not bundled. Cite Zhang et al., *Multimodal Chain-of-Thought Reasoning in Language Models*, and the dataset/model papers when using these inputs.

HotpotQA: Yang et al., EMNLP 2018. Dataset terms: CC BY-SA 4.0, https://hotpotqa.github.io/. Answer metrics follow the official implementation; its Apache 2.0 notice is retained in `licenses/HotpotQA-CODE.txt`.

ReWOO: Xu et al., *ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models*, 2023. Standard/CoT instructions and parsers, ReAct prompt/protocol conventions and evaluation structure follow `billxbf/ReWOO` commit `9cd0283043ff4be0c9d614fda2789d143ca6ffd1`. Its MIT license is retained in `licenses/ReWOO-MIT.txt`. The ReWOO Planner/Worker/Solver algorithm is not used.

ReAct: Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023. ReWOO's ReAct examples and tool protocol derive from this work (https://github.com/ysymyth/ReAct).

LangChain v0.0.187: QAEval prompt and ReAct action-parser/docstore behavior are reused/adapted without the historical dependency stack. Copyright (c) Harrison Chase. MIT license retained in `licenses/LangChain-MIT.txt`. Wikipedia transport, checkpointing and explicit usage accounting are new engineering code.
