# Multilingual GPT-2 from Scratch — with a RoPE vs. Baseline Attention Experiment

A decoder-only, GPT-2-small–class language model (**~117M parameters**) built and trained **entirely from scratch** in PyTorch, following the nanoGPT approach of Andrej Karpathy — then extended with multilingual training (**English, Hindi, Marathi**) and a controlled **Rotary Position Embedding (RoPE) vs. learned-positional-embedding** ablation.

> **▶ Live demo:** https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE  
> **⬇ Trained weights:** https://huggingface.co/yuv05/multilingual-gpt2-scratch-weights  

---

## What this project does

- Implements a decoder-only transformer from scratch: **byte-level BPE tokenizer, causal self-attention, pre-norm transformer blocks, weight tying, cosine LR schedule** — no pretrained or library model.
- Trains on **three languages / two scripts** (Latin + Devanagari), streamed and script-filtered from `allenai/c4`.
- Runs a **controlled experiment**: two otherwise-identical models, one with learned positional embeddings ("baseline") and one with **RoPE**, trained for 8,000 steps each on the same data.

## Model configuration

| | |
|---|---|
| Layers / heads / d_model | 12 / 12 / 768 |
| Vocabulary (byte-level BPE) | 32,000 |
| Context length | 512 |
| Parameters | ~117M (baseline), ~116.8M (RoPE) |
| Training tokens | ~262M per model |
| Hardware | single NVIDIA T4 (Kaggle) |

## Headline result

**RoPE converged faster and achieved a small, consistent perplexity improvement across all three languages**, while using *fewer* parameters (it replaces the learned positional table with a parameter-free rotation). Both models reached a similar final training loss.

![Training loss — baseline vs RoPE](comparison.png)

**Held-out perplexity (lower is better):**

| Language | Baseline | RoPE |
|---|---|---|
| English | 210.3 | **189.5** |
| Hindi | 6.4 | **6.1** |
| Marathi | 6.5 | **6.3** |

*Note: perplexity is comparable **within** a language (baseline vs. RoPE), not across languages — the byte-level tokenizer segments each script with different granularity, so cross-language values are not directly comparable.*

## Honest scope

This is a **base language model**, not a chatbot: it continues text rather than answering questions. Trained on ~262M tokens (~5% of GPT-2's original budget), its output is locally fluent but drifts over long spans — expected at this scale, and consistent with the educational goal of understanding the architecture end-to-end. See [`REPORT.md`](REPORT.md) for full analysis.

## Repository contents

| File | What it is |
|---|---|
| [`REPORT.md`](REPORT.md) | Full technical report — architecture, method, results, analysis |
| `train.ipynb` | The from-scratch training notebook (Kaggle) |
| `app.py` | Streamlit demo — generate text, compare both models side by side |
| `comparison.png` | Training-loss curves |
| `perplexity_results.json` | Per-language evaluation results |
| `samples/generations.md` | Selected generation samples in all three languages |

## Run the demo locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt

# download the two weight files from the Hugging Face model repo (link above)
# and place them next to app.py as:
#   model_baseline_fp16.pt
#   model_rope_fp16.pt
# plus the tokenizer files in a tokenizer/ folder

streamlit run app.py
```

Or just open the **live demo** link above — no setup required.

## Reproduce training

Open `train.ipynb` on Kaggle with a GPU, run top to bottom. It trains both models back-to-back (~8.5h on a T4) and saves all artifacts. Set `use_rope` to switch schemes, or use the combined cell that trains both.

---

*Built as an internship capstone. Inspired by Andrej Karpathy's nanoGPT.*
