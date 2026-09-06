"""
tinygpt

Author: Abraham R.

Canonical implementation of TinyGPT, shared by the ClaseIV homework notebooks:

    TP1_TinyGPT.ipynb        -> pretraining, decoding, Mixture of Experts
    TP2_TinyGPT_Finetuning   -> instruction fine-tuning + LoRA
    Serving_TinyGPT.ipynb    -> GGUF export and llama.cpp (in class)

TP-1 restates the architecture inline (reading it *is* the exercise); the others
import it from here. If you change the architecture, change it in BOTH places --
this module is the source of truth.
"""
import gc
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type

import httpx
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class MoEArgs:
    """
    Mixture of Experts hyper-parameters.

    The last two implement DeepSeekMoE (Dai et al. 2024, arXiv:2401.06066):

    - `num_shared_experts` -- experts every token passes through, always. They absorb the
      knowledge common to all tokens so the routed experts are free to specialise.
    - `expert_hidden_mult` -- the FFN expansion inside one expert. Standard is 4.0. Fine-
      grained segmentation splits each expert into `m` smaller ones: multiply `num_experts`
      and `num_experts_per_token` by `m`, divide this by `m`, and both the parameter count
      and the per-token compute are unchanged -- only the number of possible routing
      combinations grows.
    """
    num_experts: int = field(default=4)
    num_experts_per_token: int = field(default=2)
    num_shared_experts: int = field(default=0)
    expert_hidden_mult: float = field(default=4.0)


@dataclass
class GPTConfig:
    """Base configuration for TinyGPT models."""
    block_size: int = 32
    batch_size: int = 64
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    vocab_size: int = 65
    bias: bool = True
    # Share one matrix between token_emb and head. Standard since GPT-2; halves the
    # vocabulary parameters, which are almost the whole model at a large vocab.
    tie_weights: bool = True
    # Swap the feed-forward implementation without touching the model code.
    ff_class: Optional[Type[nn.Module]] = None
    moe_args: Optional[MoEArgs] = None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def load_tinyshakespeare(n_chars: Optional[int] = 200_000) -> str:
    """Downloads the tinyshakespeare corpus, optionally truncated to `n_chars`."""
    text = httpx.get(TINYSHAKESPEARE_URL, timeout=30.0).text
    return text if n_chars is None else text[:n_chars]


class CharTokenizer:
    """
    Character level tokenizer.

    Deliberately naive: one character == one token. Real tokenization (BPE,
    SentencePiece, ...) is out of scope for this course.
    """

    def __init__(self, text: str):
        self.chars: List[str] = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.special_tokens: List[str] = []

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + len(self.special_tokens)

    def add_special_tokens(self, tokens: List[str]) -> List[int]:
        """
        Appends multi-character special tokens (e.g. "<|user|>") to the end of
        the vocabulary and returns their ids.

        Appending (instead of inserting) keeps every previously learned id
        stable, so a pretrained embedding matrix only needs to *grow* -- see
        `TinyGPT.resize_token_embeddings`.
        """
        new_ids = []
        for tok in tokens:
            if tok in self.stoi:
                new_ids.append(self.stoi[tok])
                continue
            idx = self.vocab_size
            self.stoi[tok] = idx
            self.itos[idx] = tok
            self.special_tokens.append(tok)
            new_ids.append(idx)
        return new_ids

    def encode(self, s: str) -> List[int]:
        """Encodes a string, honouring any registered special tokens."""
        if not self.special_tokens:
            return [self.stoi[c] for c in s]

        ids: List[int] = []
        i = 0
        specials = sorted(self.special_tokens, key=len, reverse=True)
        while i < len(s):
            for tok in specials:
                if s.startswith(tok, i):
                    ids.append(self.stoi[tok])
                    i += len(tok)
                    break
            else:
                ids.append(self.stoi[s[i]])
                i += 1
        return ids

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class CharDataset(Dataset):
    """Next-token-prediction dataset: y is x shifted by one position."""

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Causal multi-head self-attention, batched across heads.

    All heads are projected in one matmul and attended in one batched matmul, rather than
    looping over an `AttentionHead` module per head. The loop version is easier to read but
    launches `n_head` separate small kernels per layer, and at this model size wall-clock is
    dominated by kernel launches rather than arithmetic -- so the loop cost several times
    the whole rest of the step.

    The attention is still written out by hand (scores, causal mask, softmax) rather than
    calling F.scaled_dot_product_attention, because the mask is the thing being taught and
    SDPA cannot return attention weights for the visualisation task.

    Every forward returns the same 3-tuple `(out, kv_cache, weights)`; the trailing entries
    are `None` when not requested.
    """

    def __init__(self, args: GPTConfig):
        super().__init__()
        assert args.n_embd % args.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = args.n_head
        self.head_dim = args.n_embd // args.n_head

        # One projection for all heads: (B, T, C) -> (B, T, 3C), chunked K, Q, V.
        self.key_query_value = nn.Linear(args.n_embd, 3 * args.n_embd, bias=args.bias)
        self.proj = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.dropout = nn.Dropout(args.dropout)
        self.register_buffer(
            "tril", torch.tril(torch.ones(args.block_size, args.block_size))
        )

    def _split(self, t, B, T):
        """(B, T, C) -> (B, n_head, T, head_dim)"""
        return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, x, kv_cache=None, use_cache=False, return_weights=False):
        B, T, C = x.shape
        k, q, v = self.key_query_value(x).chunk(3, dim=-1)
        k, q, v = self._split(k, B, T), self._split(q, B, T), self._split(v, B, T)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache.unbind(dim=0)      # (B, n_head, T_past, head_dim)
            k = torch.cat((k_prev, k), dim=2)
            v = torch.cat((v_prev, v), dim=2)

        new_cache = torch.stack((k, v)) if use_cache else None

        T_k = k.size(2)          # total keys attended to
        past = T_k - T           # how many of them came from the cache

        wei = q @ k.transpose(-2, -1) * (self.head_dim ** -0.5)   # (B, n_head, T, T_k)
        # Rows are offset by `past` so the mask stays correct with a KV-cache:
        # query at absolute position `past + t` may see keys 0..past + t.
        wei = wei.masked_fill(self.tril[past:T_k, :T_k] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = (wei @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.proj(out))

        # (n_head, B, T, T_k) -- the layout visualize_attention expects
        weights = wei.transpose(0, 1) if return_weights else None
        return out, new_cache, weights


# ---------------------------------------------------------------------------
# Feed-forward + Block
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Dense MLP: the module HW2 replaces with a MoE."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm transformer decoder block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)

        ff_class = config.ff_class if config.ff_class is not None else FeedForward
        self.ff = ff_class(config)

    def forward(self, x, kv_cache=None, use_cache=False, return_weights=False):
        attn_out, new_cache, weights = self.attn(
            self.ln1(x), kv_cache=kv_cache, use_cache=use_cache, return_weights=return_weights
        )
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_cache, weights


# ---------------------------------------------------------------------------
# TinyGPT
# ---------------------------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_weights:
            self.head.weight = self.token_emb.weight

    def forward(self, idx, kv_cache=None, use_cache=False, return_weights=False):
        """
        Args:
            idx:            (B, T) token ids.
            kv_cache:       list of per-layer caches from a previous call, or None.
            use_cache:      return an updated KV-cache alongside the logits.
            return_weights: return per-layer attention maps alongside the logits.

        Returns:
            `logits` when both flags are False (this is what `Trainer` expects),
            otherwise a tuple `(logits[, kv_cache][, attn_weights])`.

        Note the flags drive the return shape -- NOT whether `kv_cache` happens
        to be None. Deriving it from the input is how a cache silently never
        turns on: the very first decoding step has no cache to pass in.
        """
        B, T = idx.shape
        past_len = kv_cache[0].shape[-2] if kv_cache is not None else 0
        assert past_len + T <= self.config.block_size, (
            f"sequence of {past_len + T} exceeds block_size={self.config.block_size}"
        )

        tok_emb = self.token_emb(idx)
        # Positions must continue from where the cache left off, otherwise every
        # cached step re-uses position 0.
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = tok_emb + self.pos_emb(pos)[None, :, :]

        new_kv_cache = [] if use_cache else None
        all_weights = [] if return_weights else None

        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, updated_kv, weights = block(
                x, kv_cache=layer_cache, use_cache=use_cache, return_weights=return_weights
            )
            if use_cache:
                new_kv_cache.append(updated_kv)
            if return_weights:
                all_weights.append(weights)

        logits = self.head(self.ln_f(x))

        if not use_cache and not return_weights:
            return logits
        out = [logits]
        if use_cache:
            out.append(new_kv_cache)
        if return_weights:
            out.append(all_weights)
        return tuple(out)

    @torch.no_grad()
    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """
        Grows the input embedding and the output head to `new_vocab_size`,
        preserving every pretrained row.

        Needed when fine-tuning adds special tokens (chat/instruction markers)
        that the pretraining vocabulary did not contain.
        """
        old_vocab, n_embd = self.token_emb.weight.shape
        if new_vocab_size == old_vocab:
            return
        assert new_vocab_size > old_vocab, "shrinking the vocabulary is not supported"

        device = self.token_emb.weight.device
        dtype = self.token_emb.weight.dtype

        tied = self.head.weight is self.token_emb.weight

        new_emb = nn.Embedding(new_vocab_size, n_embd, device=device, dtype=dtype)
        new_emb.weight.normal_(mean=0.0, std=0.02)
        new_emb.weight[:old_vocab] = self.token_emb.weight
        self.token_emb = new_emb

        new_head = nn.Linear(n_embd, new_vocab_size, bias=False, device=device, dtype=dtype)
        if tied:
            new_head.weight = new_emb.weight
        else:
            new_head.weight.normal_(mean=0.0, std=0.02)
            new_head.weight[:old_vocab] = self.head.weight
        self.head = new_head

        self.config.vocab_size = new_vocab_size
        self.config.tie_weights = tied


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def trim_kv_cache(kv_cache, max_len: int):
    """Keeps only the last `max_len` positions of every layer's cache."""
    if kv_cache is None:
        return None
    return [c[..., -max_len:, :] for c in kv_cache]


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    use_cache: bool = True,
    device: Optional[str] = None,
) -> str:
    """
    Baseline decoding: sample from the full softmax at temperature 1.0.

    This is the function HW1 asks you to upgrade with greedy / temperature /
    top-k / top-p.
    """
    model.eval()
    device = device or next(model.parameters()).device
    block_size = model.config.block_size

    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)[None, :]
    kv_cache = None

    for _ in range(max_new_tokens):
        if kv_cache is None:
            idx_cond = idx[:, -block_size:]
        else:
            idx_cond = idx[:, -1:]

        out = model(idx_cond, kv_cache=kv_cache, use_cache=use_cache)
        if use_cache:
            logits, kv_cache = out
            # Never let the cache outgrow the context window.
            kv_cache = trim_kv_cache(kv_cache, block_size - 1)
        else:
            logits = out

        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

    return tokenizer.decode(idx[0].tolist())


@torch.no_grad()
def visualize_attention(model, tokenizer, prompt: str, device: Optional[str] = None):
    """Plots one heatmap per (layer, head) for a single prompt."""
    model.eval()
    device = device or next(model.parameters()).device
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)[None, :]
    idx = idx[:, -model.config.block_size:]

    _, all_weights = model(idx, return_weights=True)

    tokens = [tokenizer.itos[int(i)] for i in idx[0].tolist()]
    labels = [t.replace("\n", "\\n").replace(" ", "␣") for t in tokens]
    seq_len = len(labels)

    for layer_i, layer_w in enumerate(all_weights):
        n_heads = layer_w.shape[0]
        fig, axes = plt.subplots(1, n_heads, figsize=(4.5 * n_heads, 4.5))
        axes = [axes] if n_heads == 1 else list(axes)
        for head_i in range(n_heads):
            attn = layer_w[head_i, 0].float().cpu()  # (seq_len, seq_len)
            im = axes[head_i].imshow(attn, cmap="viridis")
            axes[head_i].set_title(f"Layer {layer_i + 1} Head {head_i + 1}")
            axes[head_i].set_xlabel("Key position")
            axes[head_i].set_ylabel("Query position")
            axes[head_i].set_xticks(range(seq_len), labels, fontsize=7)
            axes[head_i].set_yticks(range(seq_len), labels, fontsize=7)
            fig.colorbar(im, ax=axes[head_i])
        plt.tight_layout()
        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns `(total, trainable)` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_checkpoint(model: nn.Module, path: str, map_location=None) -> nn.Module:
    """
    Loads a `Trainer` checkpoint into `model`.

    Tolerates checkpoints saved from a `torch.compile`d model: compiling wraps the module
    in an `OptimizedModule`, so every state-dict key gains an `_orig_mod.` prefix. Training
    on CUDA with compile enabled and then loading those weights anywhere else fails with a
    wall of "unexpected key" errors unless the prefix is stripped.
    """
    state = torch.load(path, map_location=map_location)
    sd = state["model_state_dict"] if "model_state_dict" in state else state
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    # A tied model shares one tensor between token_emb and head, so loading a checkpoint
    # that holds two different matrices silently keeps whichever is copied last.
    tied = getattr(model, "head", None) is not None and model.head.weight is model.token_emb.weight
    if tied and {"head.weight", "token_emb.weight"} <= sd.keys():
        if not torch.equal(sd["head.weight"], sd["token_emb.weight"]):
            raise ValueError(
                f"{path} was trained without weight tying, but this model is tied. Loading "
                "it would discard the token embeddings. Retrain, or build the model with "
                "GPTConfig(..., tie_weights=False)."
            )

    model.load_state_dict(sd)
    return model


def describe(model: nn.Module, name: str = "model") -> None:
    total, trainable = count_parameters(model)
    pct = 100.0 * trainable / total if total else 0.0
    print(f"{name}: {total:,} parameters | {trainable:,} trainable ({pct:.2f}%)")


def free() -> None:
    """
    Return cached GPU blocks to the allocator.

    Call after `del`-ing a model or trainer. Neither backend releases memory on its own:
    the pool is keyed by allocation size, and the MoE's variable-shaped gathers fragment
    it badly, so a notebook that trains several models in one kernel grows without bound.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_training(trainer, epochs: int, use_amp: Optional[bool] = None,
                 dtype: torch.dtype = torch.bfloat16):
    """
    Runs `epochs` train/eval rounds and returns the loss history.

    `use_amp=None` enables mixed precision on CUDA only: MPS autocast is incomplete and
    `GradScaler` does not support that backend.
    """
    if use_amp is None:
        use_amp = trainer.device == "cuda"

    history = {"train": [], "val": []}
    print(f"mixed precision: {'on (' + str(dtype).split('.')[-1] + ')' if use_amp else 'off'}")
    for epoch in range(epochs):
        train_loss = trainer.train_model_v2(use_amp=use_amp, dtype=dtype)
        val_loss = trainer.eval_model()
        history["train"].append(float(train_loss))
        history["val"].append(float(val_loss))
        print(f"epoch {epoch + 1}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

    # Persist alongside the checkpoint so a restarted kernel can resume without retraining.
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "history.json"), "w") as fh:
            json.dump(history, fh)

    free()
    return history


def resume(model: nn.Module, save_dir: str, map_location=None):
    """Reload a finished run: weights into `model`, and its loss history back."""
    load_checkpoint(model, os.path.join(save_dir, "checkpoint_final.pt"), map_location)
    hist_path = os.path.join(save_dir, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as fh:
            return json.load(fh)
    return {"train": [], "val": []}


def plot_losses(histories: dict, title: str = "Training curves") -> None:
    """`histories` maps a run name to the dict returned by `run_training`."""
    plt.figure(figsize=(7, 4))
    for name, h in histories.items():
        epochs = range(1, len(h["train"]) + 1)
        line, = plt.plot(epochs, h["train"], marker="o", label=f"{name} train")
        plt.plot(epochs, h["val"], marker="s", linestyle="--",
                 color=line.get_color(), label=f"{name} val")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()
