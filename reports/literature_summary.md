# Literature Review — Auto Summary

![clusters](results/literature_clusters.png)

## Papers & Generated Summaries

### 1. RETA-LLM: A Retrieval-Augmented Large Language Model Toolkit

**Abstract (truncated):** Although Large Language Models (LLMs) have demonstrated extraordinary capabilities in many domains, they still have a tendency to hallucinate and generate fictitious responses to user requests. This problem can be alleviated by augmenting LLMs with…


**Generated summary:** RETA-LLM, a RETreival-Augmented LLM toolkit, is a RETreival-Augmented LLM toolkit.

---

### 2. ELIS: Efficient LLM Iterative Scheduling System with Response Length Predictor

**Abstract (truncated):** We propose ELIS, a serving system for Large Language Models (LLMs) featuring an Iterative Shortest Remaining Time First (ISRTF) scheduler designed to efficiently manage inference tasks with the shortest remaining tokens. Current LLM serving systems often…


**Generated summary:** We propose ELIS, a serving system for large language models (LLMs) featuring an Iterative Shortest Remaining Time First (ISRTF) scheduler designed to efficiently manage inference tasks with the shortest remaining tokens. Current LLM serving systems often employ a first-come-first-served scheduling strategy, which can lead to the "head-of-line blocking" problem. To overcome this limitation, it is necessary to predict LLM inference times and apply a shortest job first scheduling strategy. However, due to the auto-regressive nature of L

---

### 3. Demystifying AI Platform Design for Distributed Inference of Next-Generation LLM models

**Abstract (truncated):** Large language models (LLMs) have shown remarkable performance across a wide range of applications, often outperforming human experts. However, deploying these gigantic models efficiently for diverse inference use cases requires carefully designed hardware…


**Generated summary:** The source code is available at https://github.com/abhibambhaniya/GenZ-LLM-Analyzer. Users can also be tried it on at https://genz-llm-analyzer.streamlit.app/ without any setup on your web browser.

---

### 4. FBI-LLM: Scaling Up Fully Binarized LLMs from Scratch via Autoregressive Distillation

**Abstract (truncated):** This work presents a Fully BInarized Large Language Model (FBI-LLM), demonstrating for the first time how to train a large-scale binary language model from scratch (not the partial binary or ternary LLM like BitNet b1.58) to match the performance of its full-…


**Generated summary:** This work presents a Fully BInarized Large Language Model (FBI-LLM), demonstrating for the first time how to train a large-scale binary language model from scratch (not the partial binary or ternary LLM like BitNet b1.58) to match the performance of its full-precision counterparts (e.g., FP16 or BF16) in transformer-based LLMs. It achieves this by employing an autoregressive distillation (AD) loss with maintaining equivalent model dimensions (130M, 1.3B, 7B) and training data

---

### 5. Systematic Evaluation of LLM-as-a-Judge in LLM Alignment Tasks: Explainable Metrics and Diverse Prompt Templates

**Abstract (truncated):** LLM-as-a-Judge has been widely applied to evaluate and compare different LLM alignmnet approaches (e.g., RLHF and DPO). However, concerns regarding its reliability have emerged, due to LLM judges' biases and inconsistent decision-making. Previous research has…


**Generated summary:** Our results indicate a significant impact of prompt templates on LLM judge performance, as well as a mediocre alignment level between the tested LLM judges and human evaluators.

---

### 6. Can LLMs Lie? Investigation beyond Hallucination

**Abstract (truncated):** Large language models (LLMs) have demonstrated impressive capabilities across a variety of tasks, but their increasing autonomy in real-world applications raises concerns about their trustworthiness. While hallucinations-unintentional falsehoods-have been…


**Generated summary:** LLMs can lie? Investigation beyond Hallucination.

---

### 7. A Survey of LLM $\times$ DATA

**Abstract (truncated):** The integration of large language model (LLM) and data management (DATA) is rapidly redefining both domains. In this survey, we comprehensively review the bidirectional relationships. On the one hand, DATA4LLM, spanning large-scale data processing, storage,…


**Generated summary:** DATA4LLM feeds LLMs with high quality, diversity, and timeliness of data required for pre-training, post-training, retrieval-augmented generation, and agentic workflows:

---

### 8. Any-Precision LLM: Low-Cost Deployment of Multiple, Different-Sized LLMs

**Abstract (truncated):** Recently, considerable efforts have been directed towards compressing Large Language Models (LLMs), which showcase groundbreaking capabilities across diverse applications but entail significant deployment costs due to their large sizes. Meanwhile, much less…


**Generated summary:** emphany-precision LLM: Low-Cost Deployment of Multiple, Different-Sized LLMs.

---

### 9. Small LLMs Are Weak Tool Learners: A Multi-LLM Agent

**Abstract (truncated):** Large Language Model (LLM) agents significantly extend the capabilities of standalone LLMs, empowering them to interact with external tools (e.g., APIs, functions) and complete various tasks in a self-directed fashion. The challenge of tool use demands that…


**Generated summary:** We propose a novel approach that decomposes the aforementioned capabilities into a planner, caller, and summarizer.

---

### 10. RoleRAG: Enhancing LLM Role-Playing via Graph Guided Retrieval

**Abstract (truncated):** Large Language Models (LLMs) have shown promise in character imitation, enabling immersive and engaging conversations. However, they often generate content that is irrelevant or inconsistent with a character's background. We attribute these failures to: (1)…


**Generated summary:** RoleRAG: Enhancing LLM Role-Playing via Graph Guided Retrieval.

---

### 11. Open-LLM-Leaderboard: From Multi-choice to Open-style Questions for LLMs Evaluation, Benchmark, and Arena

**Abstract (truncated):** Multiple-choice questions (MCQ) are frequently used to assess large language models (LLMs). Typically, an LLM is given a question and selects the answer deemed most probable after adjustments for factors like length. Unfortunately, LLMs may inherently favor…


**Generated summary:** Open-LLM-Leaderboard: From Multi-choice to Open-style Questions for LLMs

---

### 12. TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices

**Abstract (truncated):** Large model inference is shifting from cloud to edge due to concerns about the privacy of user interaction data. However, edge devices often struggle with limited computing power, memory, and bandwidth, requiring collaboration across multiple devices to run…


**Generated summary:** TPI-LLM is a compute- and memory-efficient tensor parallel inference system to serve 70B-scale models.

---

### 13. Harnessing Multiple Large Language Models: A Survey on LLM Ensemble

**Abstract (truncated):** LLM Ensemble -- which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has gained substantial attention recently. The…


**Generated summary:** LLM Ensemble -- which involves the comprehensive use of multiple large language models, each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has profoundly advanced the field of LLM Ensemble.

---

### 14. ARB-LLM: Alternating Refined Binarizations for Large Language Models

**Abstract (truncated):** Large Language Models (LLMs) have greatly pushed forward advancements in natural language processing, yet their high memory and computational demands hinder practical deployment. Binarization, as an effective compression technique, can shrink model weights to…


**Generated summary:** ARB-LLM: Alternating Refined Binarizations for Large Language Models.

---

### 15. A Survey on LLM-as-a-Judge

**Abstract (truncated):** Accurate and consistent evaluation is crucial for decision-making across numerous fields, yet it remains a challenging task due to inherent subjectivity, variability, and scale. Large Language Models (LLMs) have achieved remarkable success across diverse…


**Generated summary:** We explore strategies to enhance reliability, including improving consistency, mitigating biases, and adapting to diverse assessment scenarios.

---

### 16. Strategist: Self-improvement of LLM Decision Making via Bi-Level Tree Search

**Abstract (truncated):** Traditional reinforcement learning and planning typically requires vast amounts of data and training to develop effective policies. In contrast, large language models (LLMs) exhibit strong generalization and zero-shot capabilities, but struggle with tasks…


**Generated summary:** STRATEGIST is a generalizable framework to optimize the strategy through population-based self-play simulations without the need for training data.

---

### 17. FederatedScope-LLM: A Comprehensive Package for Fine-tuning Large Language Models in Federated Learning

**Abstract (truncated):** LLMs have demonstrated great capabilities in various NLP tasks. Different entities can further improve the performance of those LLMs on their specific downstream tasks by fine-tuning LLMs. When several entities have similar interested tasks, but their data…


**Generated summary:** FS-LLM: A Comprehensive Package for Fine-Tuning Large Language Models in Federated Learning.

---

### 18. Self-Control of LLM Behaviors by Compressing Suffix Gradient into Prefix Controller

**Abstract (truncated):** We propose SelfControl, an inference-time model control method utilizing gradients to control the behavior of large language models (LLMs) without explicit human annotations. Given a desired behavior expressed in a natural language suffix string concatenated…


**Generated summary:** SelfControl is an inference-time model control method using gradients to control the behavior of large language models without explicit human annotations.

---

### 19. ODA: Observation-Driven Agent for integrating LLMs and Knowledge Graphs

**Abstract (truncated):** The integration of Large Language Models (LLMs) and knowledge graphs (KGs) has achieved remarkable success in various natural language processing tasks. However, existing methodologies that integrate LLMs and KGs often navigate the task-solving process solely…


**Generated summary:** ODA: Observation-Driven Agent for integrating LLMs and Knowledge Graphs.

---

### 20. CATP-LLM: Empowering Large Language Models for Cost-Aware Tool Planning

**Abstract (truncated):** Utilizing large language models (LLMs) for tool planning has emerged as a promising avenue for developing general AI systems, where LLMs automatically schedule external tools (e.g., vision models) to tackle complex tasks based on task descriptions. To push…


**Generated summary:** OpenCATP-LLM: Empowering Large Language Models for Cost-Aware Tool Planning.

---
