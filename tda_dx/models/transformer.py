# Transformer implementation from https://github.com/lucidrains/audiolm-pytorch/blob/main/audiolm_pytorch/audiolm_pytorch.py

from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn
import math

from tda_dx.utils import default, exists, grad_shrink, l2norm

try:
    import xformers.ops as xops

    is_xformers_available = True
except ImportError:
    is_xformers_available = False

# bias-less layernorm, being used in more recent T5s, PaLM, also in @borisdayma 's experiments shared with me
# greater stability


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# relative positional bias


class RelativePositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        layers=3
    ):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(1, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads))

    def forward(self, n, device=torch.device('cpu')):
        pos = torch.arange(n, device=device)
        rel_pos = (rearrange(pos, 'i -> i 1') - rearrange(pos, 'j -> 1 j'))
        rel_pos += (n - 1)

        # Use the dtype of the first parameter so bf16 models don't get a float32/bf16 mismatch.
        param_dtype = next(self.net[0].parameters()).dtype
        x = torch.arange(-n + 1, n, device=device).to(param_dtype)
        x = rearrange(x, '... -> ... 1')

        for layer in self.net:
            x = layer(x)

        x = x[rel_pos]
        return rearrange(x, 'i j h -> h i j')

    def forward_single_query(self, query_pos: int, total_len: int, device):
        """Bias row for one new query at absolute position `query_pos`
        attending to all `total_len` key positions (0..total_len-1).
        Returns: (heads, 1, total_len)
        """
        # relative positions: query_pos - j for j in 0..total_len-1
        rel_pos_vals = query_pos - torch.arange(total_len, device=device)  # (total_len,)
        param_dtype = next(self.net[0].parameters()).dtype
        x = rel_pos_vals.to(param_dtype).unsqueeze(-1)  # (total_len, 1)
        for layer in self.net:
            x = layer(x)
        return rearrange(x, 'j h -> h 1 j')  # (heads, 1, total_len)

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        *,
        heads,
        num_buckets=32,
        max_distance=128,
        causal=True
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.causal = causal

        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(
        self,
        n,
        device=torch.device('cpu')
    ):
        pos = torch.arange(n, device=device)
        rel_pos = (rearrange(pos, 'i -> i 1') - rearrange(pos, 'j -> 1 j'))
        rel_pos = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets, max_distance=self.max_distance)

        bias = self.relative_attention_bias(rel_pos)
        return rearrange(bias, 'i j h -> h i j')

# feedforward


class CausalDSConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ds_conv = nn.Conv1d(dim, dim, 3, bias=False, groups=dim)

    def forward(self, x):
        x = rearrange(x, 'b n c -> b c n')
        x = F.pad(x, (2, 0))
        x = self.ds_conv(x)
        return rearrange(x, 'b c n -> b n c')

    def forward_cached(self, x, past_conv_state=None):
        """KV-cache-compatible forward. Returns (out, new_conv_state).
        past_conv_state: (b, c, 2) — last 2 pre-conv channel vectors, or None for prefill.
        Always returns a non-None new_conv_state so decode steps get correct left context.
        """
        x = rearrange(x, 'b n c -> b c n')  # (b, c, n)
        if past_conv_state is not None:
            # Stateful decode: cat cached left context instead of zero-padding
            x_in = torch.cat([past_conv_state, x], dim=2)  # (b, c, 2+n)
        else:
            # Prefill: standard zero-padding (identical to original forward())
            x_in = F.pad(x, (2, 0))
        # Always save last 2 pre-conv positions so decode steps see correct context
        new_conv_state = x_in[:, :, -2:].clone()
        out = self.ds_conv(x_in)  # (b, c, n)
        return rearrange(out, 'b c n -> b n c'), new_conv_state


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def ConvFeedForward(dim, mult=4, dropout=0.1):
    inner_dim = int(dim * 2 * mult / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        CausalDSConv(inner_dim * 2),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False)
    )

def FeedForward(dim, mult=4, dropout=0.1):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False)
    )

# attention


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        non_causal_prefix=0,
        dim_head=128,
        dim_context=None,
        heads=8,
        num_kv_heads=1,
        norm_context=False,
        num_null_kv=0,
        dropout=0.1,
        scale=8,
        use_memory_efficient_attention=False,
        use_xsa=False
    ):
        super().__init__()
        self.heads = heads
        self.num_kv_heads = num_kv_heads
        assert heads % num_kv_heads == 0, f'heads ({heads}) must be divisible by num_kv_heads ({num_kv_heads})'
        self.scale = scale
        self.causal = causal
        self.non_causal_prefix = non_causal_prefix
        self.dropout = dropout
        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.use_xsa = use_xsa
        if self.use_memory_efficient_attention and not is_xformers_available:
            raise ImportError("Please install xformers to use memory efficient attention")

        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout) if not self.use_memory_efficient_attention else None

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(2, num_null_kv, dim_head)) if num_null_kv > 0 else None

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_head * num_kv_heads * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context=None,
        mask=None,
        attn_bias=None,
        prefix_context=None,
        prefix_context_mask=None
    ):
        b, n, _, device = *x.shape, x.device

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        # take care of prefix-based self attention conditioning
        # make sure to either concat the to the self attention mask or lengthen it accordingly

        if exists(prefix_context):
            kv_input = torch.cat((prefix_context, kv_input), dim=-2)
            prefix_seq_len = prefix_context.shape[-2]

            if not exists(mask):
                mask = torch.ones((b, n), device=device, dtype=torch.bool)

            if exists(prefix_context_mask):
                mask = torch.cat((prefix_context_mask, mask), dim=-1)
            else:
                mask = F.pad(mask, (prefix_seq_len, 0), value=True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (prefix_seq_len, 0), value=0.)

        # prenorm

        x = self.norm(x)

        # project for queries, keys, values

        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # split into heads: Q → all heads, K/V → kv_heads (GQA)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_kv_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_kv_heads)

        # expand KV heads to match Q heads (GQA/MQA -> full heads) before saving self V

        if self.num_kv_heads < self.heads:
            repeat_factor = self.heads // self.num_kv_heads
            k = repeat(k, 'b h n d -> b (h r) n d', r=repeat_factor)
            v = repeat(v, 'b h n d -> b (h r) n d', r=repeat_factor)

        # save the value vectors aligned with the current query positions for optional XSA

        self_v = None
        is_self_attention = not exists(context)
        if self.use_xsa and is_self_attention:
            self_v = v[..., -n:, :]

        # null key / values

        if self.num_null_kv > 0:
            null_k, null_v = repeat(self.null_kv, 'kv n d -> kv b h n d', b=b, h=self.heads).unbind(dim=0)
            k = torch.cat((null_k, k), dim=-2)
            v = torch.cat((null_v, v), dim=-2)

        # QK normalization (arXiv:2302.05442)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # attention

        if self.use_memory_efficient_attention:
            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.)

                if exists(mask):
                    mask = F.pad(mask, (self.num_null_kv, 0), value=True)
                    mask = rearrange(mask, 'b j -> b 1 1 j')
                    attn_bias = attn_bias.masked_fill(~mask, -torch.finfo(attn_bias.dtype).max)

                if self.causal:
                    i, j = attn_bias.shape[-2:]
                    causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)

                    if self.non_causal_prefix > 0:
                        causal_mask[:self.non_causal_prefix, :(self.non_causal_prefix + j - i)] = False

                    attn_bias = attn_bias.masked_fill(causal_mask, -torch.finfo(attn_bias.dtype).max)

            q = rearrange(q, 'b h n d -> b n h d')
            k = rearrange(k, 'b h n d -> b n h d')
            v = rearrange(v, 'b h n d -> b n h d')

            # compute attention
            out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=self.dropout)

            out = rearrange(out, 'b n h d -> b h n d')

        else:
            sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.)
                sim = sim + attn_bias

            if exists(mask):
                mask = F.pad(mask, (self.num_null_kv, 0), value=True)
                mask = rearrange(mask, 'b j -> b 1 1 j')
                sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

            if self.causal:
                i, j = sim.shape[-2:]
                causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(j - i + 1)

                if self.non_causal_prefix > 0:
                    causal_mask[:self.non_causal_prefix, :(self.non_causal_prefix + j - i)] = False

                sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim=-1)
            attn = self.attn_dropout(attn)

            # aggregate
            out = einsum('b h i j, b h j d -> b h i d', attn, v)

        if exists(self_v):
            self_v = F.normalize(self_v, dim=-1)
            out = out - (out * self_v).sum(dim=-1, keepdim=True) * self_v

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

    def forward_cached(self, x, past_kv=None, rel_pos_bias=None, mask=None):
        """Inference-only cached self-attention (no context/prefix_context support).
        past_kv: tuple(k_cache, v_cache) each (b, num_kv_heads, past_n, d_head), or None for prefill.
        Returns: (out, (new_k, new_v)) where new_k/v are at num_kv_heads resolution (pre-GQA-repeat).
        """
        b, n, _, device = *x.shape, x.device

        # Mirrors Attention.forward(): kv_input = x (un-normed), q from norm(x)
        # This matches: kv_input = default(context, x) captured BEFORE x = self.norm(x)
        kv_input = x
        q = self.to_q(self.norm(x))
        k_new, v_new = self.to_kv(kv_input).chunk(2, dim=-1)

        q    = rearrange(q,     'b n (h d) -> b h n d', h=self.heads)
        k_new = rearrange(k_new, 'b n (h d) -> b h n d', h=self.num_kv_heads)
        v_new = rearrange(v_new, 'b n (h d) -> b h n d', h=self.num_kv_heads)

        # QK normalisation (same as in forward(), before caching)
        q, k_new = map(l2norm, (q, k_new))
        q    = q    * self.q_scale
        k_new = k_new * self.k_scale

        # Extend cache
        if past_kv is not None:
            k = torch.cat([past_kv[0], k_new], dim=2)
            v = torch.cat([past_kv[1], v_new], dim=2)
        else:
            k, v = k_new, v_new

        new_kv = (k, v)  # saved pre-GQA-repeat (num_kv_heads resolution)

        # GQA expand
        if self.num_kv_heads < self.heads:
            r = self.heads // self.num_kv_heads
            k = repeat(k, 'b h n d -> b (h r) n d', r=r)
            v = repeat(v, 'b h n d -> b (h r) n d', r=r)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # Causal mask only for multi-query prefill; single-token decode never needs it
        # (all past K/V are causally valid by construction)
        if self.causal and n > 1:
            i, j = n, k.shape[2]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        out  = einsum('b h i j, b h j d -> b h i d', attn, v)

        # XSA: subtract self-value component — mirrors Attention.forward()
        if self.use_xsa:
            self_v = F.normalize(v[..., -n:, :], dim=-1)  # new positions' values
            out = out - (out * self_v).sum(dim=-1, keepdim=True) * self_v

        out  = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), new_kv

# transformer


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        dim_context=None,
        cross_attend=False,
        attn_dropout=0.,
        ff_dropout=0.,
        use_conv_ff=True,
        grad_shrink_alpha=0.1,
        cond_as_self_attn_prefix=False,
        non_causal_prefix_size=0,
        relative_position_bias_type='continuous',
        **kwargs
    ):
        super().__init__()
        assert not (cross_attend and cond_as_self_attn_prefix)
        self.dim_context = default(dim_context, dim)

        self.cond_as_self_attn_prefix = cond_as_self_attn_prefix

        self.grad_shrink = partial(grad_shrink, alpha=grad_shrink_alpha)

        self.layers = nn.ModuleList([])

        if relative_position_bias_type == 'continuous':
            self.rel_pos_bias = RelativePositionBias(dim=dim // 2, heads=heads)
        elif relative_position_bias_type == 't5':
            self.rel_pos_bias = T5RelativePositionBias(heads=heads, num_buckets=32, max_distance=128)
        elif relative_position_bias_type == 'none':
            self.rel_pos_bias = None
        else:
            raise ValueError(f'invalid relative position bias type: {relative_position_bias_type}')

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dropout=attn_dropout, causal=True, non_causal_prefix=non_causal_prefix_size, **kwargs),
                Attention(dim=dim, heads=heads, dropout=attn_dropout, dim_context=dim_context,
                          num_null_kv=1, norm_context=True, **kwargs) if cross_attend else None,
                ConvFeedForward(dim=dim, dropout=ff_dropout) if use_conv_ff else FeedForward(dim=dim, dropout=ff_dropout),
            ]))

        self.norm = LayerNorm(dim)
        self._gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        """Trade activation memory for recomputation (~1.3x slower, mathematically identical)."""
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self._gradient_checkpointing = False

    def _run_block(self, attn, cross_attn, ff, x, rel_pos_bias, self_attn_mask, self_attn_kwargs, context, context_mask):
        """Single transformer block forward used with or without gradient checkpointing."""
        x = attn(x, attn_bias=rel_pos_bias, mask=self_attn_mask, **self_attn_kwargs) + x
        if exists(cross_attn):
            x = cross_attn(x, context=context, mask=context_mask) + x
        x = ff(x) + x
        return x

    def forward(
        self,
        x,
        self_attn_mask=None,
        context=None,
        context_mask=None,
        attn_bias=None,
    ):
        assert not (self.cond_as_self_attn_prefix and not exists(context))
        assert not (exists(
            context) and context.shape[-1] != self.dim_context), f'you had specified a conditioning dimension of {self.dim_context}, yet what was received by the transformer has dimension of {context.shape[-1]}'

        n, device = x.shape[1], x.device

        # from cogview paper, adopted by GLM 130B LLM, decreases likelihood of attention net instability
        x = self.grad_shrink(x)

        if exists(attn_bias):
            rel_pos_bias = attn_bias
        else:
            rel_pos_bias = self.rel_pos_bias(n, device = device) if exists(self.rel_pos_bias) else None

        self_attn_kwargs = dict()
        if self.cond_as_self_attn_prefix:
            self_attn_kwargs = dict(
                prefix_context=context,
                prefix_context_mask=context_mask
            )

        use_ckpt = self._gradient_checkpointing and torch.is_grad_enabled()

        for attn, cross_attn, ff in self.layers:
            if use_ckpt:
                import torch.utils.checkpoint as ckpt_util
                x = ckpt_util.checkpoint(
                    self._run_block,
                    attn, cross_attn, ff,
                    x, rel_pos_bias, self_attn_mask, self_attn_kwargs,
                    context, context_mask,
                    use_reentrant=False,
                )
            else:
                x = self._run_block(attn, cross_attn, ff, x, rel_pos_bias, self_attn_mask, self_attn_kwargs, context, context_mask)

        return self.norm(x)

    def forward_cached(self, x, past_kv_list=None, past_conv_list=None):
        """Inference-only KV+conv-state cached forward. Never used during training.

        x: (b, n, dim) — new token(s) to process (n=1 for decode, n=full_len for prefill)
        past_kv_list: List[Tuple[k,v]] of length depth, or None for prefill.
        past_conv_list: List[Tensor(b,c,2) or None] of length depth, or None for prefill.

        Returns: (hidden, new_kv_list, new_conv_list)
        """
        n, device = x.shape[1], x.device
        past_n = past_kv_list[0][0].shape[2] if past_kv_list is not None else 0
        total_n = past_n + n

        x = self.grad_shrink(x)

        # Compute relative position bias
        if exists(self.rel_pos_bias):
            if n > 1:
                # Prefill: full (n × n) bias matrix (same as regular forward)
                rel_pos_bias = self.rel_pos_bias(n, device=device)
            else:
                # Decode: single new query at absolute position `past_n`
                rel_pos_bias = self.rel_pos_bias.forward_single_query(past_n, total_n, device)
        else:
            rel_pos_bias = None

        new_kv_list   = []
        new_conv_list = []

        for layer_idx, (attn, cross_attn, ff) in enumerate(self.layers):
            past_kv   = past_kv_list[layer_idx]   if past_kv_list   is not None else None
            past_conv = past_conv_list[layer_idx]  if past_conv_list is not None else None

            # Self-attention (cached)
            attn_out, new_kv = attn.forward_cached(x, past_kv=past_kv, rel_pos_bias=rel_pos_bias)
            x = x + attn_out
            new_kv_list.append(new_kv)

            # Feed-forward
            if isinstance(ff[2], CausalDSConv):
                # ConvFeedForward path: thread conv state through CausalDSConv
                y = ff[1](ff[0](x))                        # LayerNorm → Linear
                y, new_conv = ff[2].forward_cached(y, past_conv)  # CausalDSConv with state
                y = ff[3](y)                               # GEGLU
                y = ff[6](ff[5](ff[4](y)))                 # LayerNorm → Dropout → Linear
                x = x + y
            else:
                # Plain FeedForward (no conv state)
                x = x + ff(x)
                new_conv = None
            new_conv_list.append(new_conv)

        return self.norm(x), new_kv_list, new_conv_list
