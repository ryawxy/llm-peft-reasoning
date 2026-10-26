# Adapting and Reasoning with Small LLMs

Three experiments on getting more out of small language models (0.5B–0.6B parameters):

1. **Parameter-efficient fine-tuning:** QLoRA adapters, rank ablation, and specialist vs. general adapters.
2. **Inference-time reasoning:** six prompting and decoding strategies compared on GSM8K.
3. **Retrieval-augmented generation:** a RAG pipeline built from scratch, extended with conversational memory.

## Contents

| Notebook | Topic | Models | Data |
|---|---|---|---|
| `peft/qlora_adapters.ipynb` | QLoRA / LoRA fine-tuning | Qwen2.5-0.5B-Instruct | GSM8K, CommonsenseQA, AG News |
| `reasoning/inference_time_reasoning.ipynb` | Inference-time reasoning | Qwen3-0.6B | GSM8K (64-problem subset) |
| `rag/conversational_rag.ipynb` | RAG from scratch + memory | MiniLM-L6, GPT-2, FLAN-T5 | SQuAD v1 |

## 1. QLoRA adapters

**LoRA rank ablation (CommonsenseQA, 300 training examples, exact match)**

| Rank | r = 4 | r = 16 | r = 64 |
|---|---|---|---|
| Val. accuracy | 0.59 | **0.62** | 0.53 |

Using r = 16, I trained one specialist adapter per task. I also trained a general adapter on all three tasks with a **matched optimizer budget** (80 updates), so the specialist-vs-general comparison is fair. Specialist in-domain accuracy:

| Task | GSM8K | CommonsenseQA | AG News |
|---|---|---|---|
| Specialist adapter | 0.30 | 0.62 | 0.85 |

The final evaluation is a transfer matrix: the base model and all four adapters, each evaluated on all three tasks.

## 2. Inference-time reasoning (Qwen3-0.6B on GSM8K)

| Method | Accuracy | Avg. tokens | Latency / problem |
|---|---|---|---|
| Direct answer (`/no_think`) | 4.7% | 12 | 0.09 s |
| Chain-of-thought (`/no_think`) | 59.4% | 212 | 1.37 s |
| Chain-of-thought (`/think`, 512-token cap) | 50.0% | 512 | 3.35 s |
| Chain-of-thought (`/think`, 1024-token cap) | **64.1%** | 1011 | 6.64 s |
| Self-consistency (N = 5) | 53.1% | 1015 | 7.06 s |
| Few-shot (k = 1) | 39.1% | 176 | 1.17 s |
| Self-refinement | 60.9% | 452 | 2.97 s |

**Takeaways**
- Chain-of-thought gives the biggest gain per token.
- Thinking mode helps only when its token budget is large enough. With a 512-token cap, every trace was truncated before the model wrote an answer.
- On a 64-problem set, self-consistency and few-shot prompting did not beat single chain-of-thought.

## 3. Conversational RAG

- The retrieval corpus is SQuAD v1 contexts, chunked and embedded with `all-MiniLM-L6-v2` and indexed in FAISS.
- The first version is a stateless pipeline with a GPT-2 generator. It fails on follow-up questions such as *"Where was it played?"*
- A history-aware query-rewriting step turns each follow-up into a standalone question before retrieval, and FLAN-T5 generates the answer.
- I also compare decoder-only (GPT-2) and encoder-decoder (FLAN-T5) generators on the same retrieved context.

## Setup

```bash
pip install torch transformers accelerate peft bitsandbytes datasets sentence-transformers faiss-cpu langchain-community langchain-huggingface pandas matplotlib
```

Every notebook runs on a single free-tier GPU (Colab or Kaggle T4).

