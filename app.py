"""
Multilingual GPT-2 (from scratch) — interactive demo
Base GPT-2-style model trained from scratch on English / Hindi / Marathi,
with a baseline vs. RoPE positional-encoding comparison.
Weights + tokenizer are pulled from the Hugging Face model repo at runtime.
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from tokenizers import ByteLevelBPETokenizer
from huggingface_hub import hf_hub_download

HF_REPO = "yuv05/multilingual-gpt2-scratch-weights"

CONFIG = dict(
    vocab_size=32000, block_size=512, n_layer=12, n_head=12,
    n_embd=768, dropout=0.1, bias=True, use_rope=False,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def forward(self, seq_len):
        return self.cos[:, :, :seq_len, :], self.sin[:, :, :seq_len, :]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.n_embd = cfg["n_head"], cfg["n_embd"]
        self.head_dim = self.n_embd // self.n_head
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=cfg["bias"])
        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=cfg["bias"])
        self.attn_drop = nn.Dropout(cfg["dropout"])
        self.resid_drop = nn.Dropout(cfg["dropout"])
        self.use_rope = cfg["use_rope"]
        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim, cfg["block_size"])

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_rope:
            cos, sin = self.rope(T)
            q, k = apply_rope(q, k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"], bias=cfg["bias"])
        self.proj = nn.Linear(4 * cfg["n_embd"], cfg["n_embd"], bias=cfg["bias"])
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_rope = cfg["use_rope"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        if not self.use_rope:
            self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        self.tok_emb.weight = self.head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if not self.use_rope:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


@st.cache_resource(show_spinner="Loading tokenizer...")
def load_tokenizer():
    vocab = hf_hub_download(repo_id=HF_REPO, filename="tokenizer/vocab.json")
    merges = hf_hub_download(repo_id=HF_REPO, filename="tokenizer/merges.txt")
    return ByteLevelBPETokenizer(vocab, merges)


@st.cache_resource(show_spinner="Loading model...")
def load_model(use_rope):
    cfg = dict(CONFIG)
    cfg["use_rope"] = use_rope
    model = GPT(cfg).to(DEVICE)
    fname = "model_rope_fp16.pt" if use_rope else "model_baseline_fp16.pt"
    path = hf_hub_download(repo_id=HF_REPO, filename=fname)
    ck = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    if DEVICE == "cpu":
        model = model.float()
    model.eval()
    return model, ck.get("step", None)


@st.cache_data(show_spinner=False)
def load_perplexity():
    if os.path.exists("perplexity_results.json"):
        with open("perplexity_results.json") as f:
            return json.load(f)
    return None


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=80, temperature=0.8, top_k=40):
    ids = tok.encode(prompt).ids
    if len(ids) == 0:
        ids = [tok.token_to_id("<|endoftext|>")]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    for _ in range(max_new_tokens):
        x_cond = x[:, -CONFIG["block_size"]:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        x = torch.cat([x, nxt], dim=1)
    return tok.decode(x[0].tolist())


st.set_page_config(page_title="Multilingual GPT-2 - from scratch", page_icon="_", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,900&family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{ --paper:#0f1419; --paper-2:#161d26; --panel:#1b232e; --line:#2a3644;
  --ink:#e8edf2; --ink-soft:#9fb0c0; --ink-faint:#61728a; --baseline:#e0a458; --rope:#4fd1c5; }
.stApp{ background:radial-gradient(1200px 600px at 80% -10%, #1a2430 0%, transparent 60%), var(--paper); }
#MainMenu, footer, header{ visibility:hidden; }
.block-container{ padding-top:2.2rem; max-width:1180px; }
.masthead{ border:1px solid var(--line); border-radius:18px; background:linear-gradient(180deg, var(--paper-2), var(--paper)); padding:26px 30px 22px; position:relative; overflow:hidden; }
.masthead:before{ content:""; position:absolute; inset:0; background-image:linear-gradient(var(--line) 1px, transparent 1px); background-size:100% 26px; opacity:.18; pointer-events:none; }
.eyebrow{ font-family:'JetBrains Mono',monospace; font-size:.72rem; letter-spacing:.32em; text-transform:uppercase; color:var(--ink-faint); display:flex; gap:14px; align-items:center; }
.eyebrow .dot{ width:7px;height:7px;border-radius:50%;background:var(--rope); box-shadow:0 0 10px var(--rope); display:inline-block; }
.title{ font-family:'Fraunces',serif; font-weight:900; color:var(--ink); font-size:clamp(2.1rem,4.5vw,3.4rem); line-height:1.02; letter-spacing:-.02em; margin:14px 0 6px; }
.title .rope{ color:var(--rope); font-style:italic; }
.subtitle{ font-family:'Space Grotesk',sans-serif; color:var(--ink-soft); font-size:1.02rem; max-width:56ch; line-height:1.5; }
.spec-row{ display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }
.chip{ font-family:'JetBrains Mono',monospace; font-size:.74rem; color:var(--ink-soft); background:var(--panel); border:1px solid var(--line); border-radius:999px; padding:5px 13px; }
.chip b{ color:var(--ink); font-weight:500; }
.seclabel{ font-family:'JetBrains Mono',monospace; font-size:.72rem; letter-spacing:.28em; text-transform:uppercase; color:var(--ink-faint); margin:34px 0 12px; display:flex; align-items:center; gap:12px; }
.seclabel:after{ content:""; flex:1; height:1px; background:var(--line); }
.modelcard{ border:1px solid var(--line); border-radius:14px; background:var(--panel); padding:16px 18px; height:100%; }
.modelcard .name{ font-family:'Fraunces',serif; font-weight:600; font-size:1.15rem; color:var(--ink); }
.modelcard .meta{ font-family:'JetBrains Mono',monospace; font-size:.76rem; color:var(--ink-faint); margin-top:4px; }
.tag-base{ color:var(--baseline); } .tag-rope{ color:var(--rope); }
.gen-out{ border:1px solid var(--line); border-left:3px solid var(--rope); border-radius:12px; background:var(--paper-2); padding:20px 22px; margin-top:10px; font-family:'Space Grotesk',sans-serif; font-size:1.05rem; line-height:1.62; color:var(--ink); white-space:pre-wrap; word-break:break-word; }
.gen-out.base{ border-left-color:var(--baseline); }
.gen-prompt{ color:var(--ink-faint); } .gen-cont{ color:var(--ink); }
.stButton>button{ font-family:'Space Grotesk',sans-serif; font-weight:600; background:var(--rope); color:#06201d; border:0; border-radius:10px; padding:.55rem 1.3rem; }
.stButton>button:hover{ filter:brightness(1.08); transform:translateY(-1px); }
.stTextArea textarea{ background:var(--paper-2)!important; color:var(--ink)!important; border:1px solid var(--line)!important; border-radius:12px!important; font-family:'Space Grotesk',sans-serif!important; font-size:1.05rem!important; }
.stSlider label, .stTextArea label, .stRadio label{ font-family:'JetBrains Mono',monospace!important; font-size:.76rem!important; letter-spacing:.06em; color:var(--ink-soft)!important; }
.pxtable{ width:100%; border-collapse:collapse; font-family:'JetBrains Mono',monospace; }
.pxtable th{ text-align:left; font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; color:var(--ink-faint); font-weight:500; padding:10px 14px; border-bottom:1px solid var(--line); }
.pxtable td{ padding:12px 14px; border-bottom:1px solid var(--line); color:var(--ink); font-size:.92rem; }
.pxtable .lang{ color:var(--ink-soft); } .win{ color:var(--rope); font-weight:500; }
.pxbar{ height:6px; border-radius:3px; background:var(--line); position:relative; overflow:hidden; min-width:80px; }
.pxbar i{ position:absolute; left:0; top:0; bottom:0; border-radius:3px; }
.note{ font-family:'Space Grotesk',sans-serif; color:var(--ink-faint); font-size:.85rem; line-height:1.5; margin-top:10px; border-left:2px solid var(--line); padding-left:12px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<div class="masthead">
  <div class="eyebrow"><span class="dot"></span> decoder-only &middot; trained from scratch &middot; nanoGPT lineage</div>
  <div class="title">A small GPT-2 that reads<br>English, &#2361;&#2367;&#2306;&#2342;&#2368; &amp; <span class="rope">&#2350;&#2352;&#2366;&#2336;&#2368;</span></div>
  <div class="subtitle">A ~117M-parameter transformer built and trained end-to-end on Kaggle - with a controlled experiment swapping learned positional embeddings for <b>Rotary Position Embeddings (RoPE)</b> inside the attention block.</div>
  <div class="spec-row">
    <span class="chip"><b>12</b> layers</span><span class="chip"><b>12</b> heads</span>
    <span class="chip"><b>768</b> d_model</span><span class="chip"><b>32k</b> BPE vocab</span>
    <span class="chip"><b>512</b> ctx</span><span class="chip">3 languages &middot; 2 scripts</span>
  </div>
</div>
""", unsafe_allow_html=True)

tok = load_tokenizer()

st.markdown('<div class="seclabel">01 &middot; Generate</div>', unsafe_allow_html=True)

QUICK = ["India is a large and diverse country where people speak many", "सुबह के समय सूरज पूर्व दिशा से निकलता है और", "मुंबई हे शहर आपल्या सुंदरतेसाठी आणि व्यस्त जीवनासाठी",
         "The city of Mumbai is famous for its", "महाराष्ट्र हे भारतातील एक महत्त्वाचे राज्य असून तेथील लोक"]

if "prompt" not in st.session_state:
    st.session_state.prompt = "The capital of India is"

# handle quick-prompt clicks BEFORE the text_area widget is created,
# so we can legally update session_state.prompt
def _pick(q):
    st.session_state.prompt = q

left, right = st.columns([3, 2], gap="large")
with left:
    prompt = st.text_area("PROMPT - english / हिंदी / मराठी", key="prompt", height=120)
    st.markdown("<span style='color:#61728a;font-family:monospace;font-size:.72rem;letter-spacing:.06em'>QUICK PROMPTS (click to fill)</span>", unsafe_allow_html=True)
    qcols = st.columns(len(QUICK))
    for i, q in enumerate(QUICK):
        qcols[i].button(q, key=f"qp_{i}", on_click=_pick, args=(q,))
with right:
    mode = st.radio("MODEL", ["Baseline", "RoPE", "Compare both"], index=2)
    temperature = st.slider("TEMPERATURE", 0.1, 1.5, 0.65, 0.05)
    top_k = st.slider("TOP-K", 1, 100, 25, 1)
    max_new = st.slider("MAX NEW TOKENS", 20, 200, 80, 10)

go = st.button("Generate", use_container_width=True)


def render_output(text, prompt_str, kind):
    cont = text[len(prompt_str):] if text.startswith(prompt_str) else text
    cls = "base" if kind == "baseline" else ""
    st.markdown(
        f'<div class="gen-out {cls}"><span class="gen-prompt">{prompt_str}</span>'
        f'<span class="gen-cont">{cont}</span></div>', unsafe_allow_html=True)


if go:
    final_prompt = prompt.strip() or QUICK[0]
    if mode == "Compare both":
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="modelcard"><span class="name">Baseline</span>'
                        '<div class="meta tag-base">learned positional embeddings</div></div>', unsafe_allow_html=True)
            with st.spinner("baseline generating..."):
                m, _ = load_model(False)
                out = generate(m, tok, final_prompt, max_new, temperature, top_k)
            render_output(out, final_prompt, "baseline")
        with c2:
            st.markdown('<div class="modelcard"><span class="name">RoPE</span>'
                        '<div class="meta tag-rope">rotary position embeddings</div></div>', unsafe_allow_html=True)
            with st.spinner("rope generating..."):
                m, _ = load_model(True)
                out = generate(m, tok, final_prompt, max_new, temperature, top_k)
            render_output(out, final_prompt, "rope")
    else:
        use_rope = (mode == "RoPE")
        with st.spinner("generating..."):
            m, _ = load_model(use_rope)
            out = generate(m, tok, final_prompt, max_new, temperature, top_k)
        render_output(out, final_prompt, "rope" if use_rope else "baseline")

px = load_perplexity()
if px or os.path.exists("comparison.png"):
    st.markdown('<div class="seclabel">02 &middot; The experiment - baseline vs RoPE</div>', unsafe_allow_html=True)
    cc1, cc2 = st.columns([1, 1], gap="large")
    with cc1:
        if px:
            rows = ""
            for lang, d in px.items():
                bppl = d.get("baseline_ppl"); rppl = d.get("rope_ppl")
                win = "rope" if (rppl is not None and bppl is not None and rppl < bppl) else "base"
                mx = max(bppl or 1, rppl or 1)
                bw = int(100 * (bppl or 0) / mx); rw = int(100 * (rppl or 0) / mx)
                rows += (
                    f'<tr><td class="lang">{lang}</td>'
                    f'<td>{bppl:.1f}<div class="pxbar"><i style="width:{bw}%;background:var(--baseline)"></i></div></td>'
                    f'<td class="{"win" if win=="rope" else ""}">{rppl:.1f}<div class="pxbar"><i style="width:{rw}%;background:var(--rope)"></i></div></td>'
                    f'</tr>'
                )
            st.markdown(
                '<table class="pxtable">'
                '<tr><th>lang</th><th>baseline&nbsp;ppl</th><th>rope&nbsp;ppl</th></tr>'
                f'{rows}</table>'
                '<div class="note">Lower is better. Compare within a language (baseline vs RoPE) - '
                'cross-language values are not directly comparable because tokenization granularity differs per script.</div>',
                unsafe_allow_html=True)
    with cc2:
        if os.path.exists("comparison.png"):
            st.image("comparison.png", caption="Training loss (smoothed) - baseline vs RoPE", use_container_width=True)

st.markdown('<div class="seclabel">notes</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="note">This is a <b>base language model</b>, not a chatbot: it continues text rather than '
    "answering questions. Trained on ~262M tokens (~5% of GPT-2's budget), so output is locally fluent "
    'but not long-range coherent - expected at this scale.</div>', unsafe_allow_html=True)