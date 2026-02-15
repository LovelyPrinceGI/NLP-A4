import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Helper functions
# -----------------------------

def get_attn_pad_mask(seq_q, seq_k, pad_id):
    """
    seq_q: [B, Lq], seq_k: [B, Lk]
    return: [B, Lq, Lk] mask where positions of PAD in seq_k are True.
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.eq(pad_id).unsqueeze(1)  # [B, 1, Lk]
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def build_sentence_batch(input_ids_batch, cls_id, sep_id):
    """
    input_ids_batch: [B, L] (token ids without CLS/SEP)
    return:
      input_ids:   [B, L+2] with [CLS] ... [SEP]
      segment_ids: [B, L+2] all zeros (single sentence)
    """
    batch_size, seq_len = input_ids_batch.size()
    device = input_ids_batch.device

    cls = torch.full((batch_size, 1), cls_id, dtype=torch.long, device=device)
    sep = torch.full((batch_size, 1), sep_id, dtype=torch.long, device=device)

    input_ids = torch.cat([cls, input_ids_batch, sep], dim=1)
    segment_ids = torch.zeros_like(input_ids, dtype=torch.long)

    return input_ids, segment_ids


def mean_pool(hidden_states, input_ids, pad_id):
    """
    hidden_states: [B, L, D]
    input_ids:     [B, L]  (with PAD tokens)
    pad_id:        int
    return:        [B, D] mean-pooled embeddings over non-PAD tokens
    """
    mask = (input_ids != pad_id).unsqueeze(-1)  # [B, L, 1]
    masked_hidden = hidden_states * mask
    sum_hidden = masked_hidden.sum(dim=1)               # [B, D]
    lengths = mask.sum(dim=1).clamp(min=1)              # [B, 1]
    mean_hidden = sum_hidden / lengths                  # [B, D]
    return mean_hidden


# -----------------------------
# Core model components
# -----------------------------

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_segments):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        bsz, seq_len = x.size()
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)
        emb = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(emb)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q: [B, H, Lq, d_k], K: [B, H, Lk, d_k], V: [B, H, Lk, d_v]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Q.size(-1))
        scores.masked_fill_(attn_mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attn = None
        self.sdpa = ScaledDotProductAttention()

    def forward(self, Q, K, V, attn_mask):
        residual = Q
        batch_size = Q.size(0)

        # [B, L, H*d_k] -> [B, H, L, d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # attn_mask: [B, Lq, Lk] -> [B, H, Lq, Lk]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attn = self.sdpa(q_s, k_s, v_s, attn_mask)
        self.attn = attn

        # [B, H, L, d_v] -> [B, L, H*d_v]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v
        )
        output = self.fc(context)
        return self.layer_norm(output + residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.layer_norm(x + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        out = self.self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        out = self.pos_ffn(out)
        return out


class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, d_k, d_v, d_ff,
                 n_layers, n_heads, max_len, n_segments, pad_id):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, max_len, n_segments)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_k, d_v, d_ff, n_heads)
            for _ in range(n_layers)
        ])

        # NSP head
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.classifier = nn.Linear(d_model, 2)

        # MLM head
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # decoder shares weight with token embedding
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

        # store pad_id for attention masks and pooling
        self.pad_id = pad_id

    def encode(self, input_ids, segment_ids):
        """
        Return encoder hidden states without applying MLM/NSP heads.
        """
        x = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, pad_id=self.pad_id)
        for layer in self.layers:
            x = layer(x, enc_self_attn_mask)
        return x  # [B, L, d_model]

    def forward(self, input_ids, segment_ids, masked_pos):
        """
        Standard BERT forward: returns MLM and NSP logits.
        """
        x = self.encode(input_ids, segment_ids)

        # NSP using [CLS]
        pooled = self.activ(self.fc(x[:, 0]))
        logits_nsp = self.classifier(pooled)

        # MLM using masked positions
        masked_pos = masked_pos.unsqueeze(-1).expand(-1, -1, x.size(-1))
        h_masked = torch.gather(x, 1, masked_pos)
        h_masked = self.norm(F.gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias

        return logits_lm, logits_nsp


class SentenceEncoder(nn.Module):
    """
    Wrapper to obtain sentence embeddings from a pretrained BERT encoder.
    """

    def __init__(self, bert_model, cls_id, sep_id, pad_id):
        super().__init__()
        self.bert = bert_model
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id

    def forward(self, input_ids_batch):
        # add [CLS]/[SEP] and build segment ids
        input_ids, segment_ids = build_sentence_batch(
            input_ids_batch, self.cls_id, self.sep_id
        )

        # use BERT encoder to get contextualized token representations
        hidden_states = self.bert.encode(input_ids, segment_ids)  # [B, L, D]

        # compute mean-pooled sentence embedding
        sentence_embeddings = mean_pool(hidden_states, input_ids, pad_id=self.pad_id)
        return sentence_embeddings


class SBERTClassifier(nn.Module):
    """
    SBERT-style classifier: [u; v; |u - v|] -> linear -> logits(3).
    """

    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(3 * hidden_dim, 3)

    def forward(self, input_ids_a, input_ids_b):
        # sentence embeddings for premise and hypothesis
        u = self.encoder(input_ids_a)  # [B, D]
        v = self.encoder(input_ids_b)  # [B, D]

        # absolute difference
        uv_abs = torch.abs(u - v)

        # concatenation [u; v; |u - v|]
        x = torch.cat([u, v, uv_abs], dim=-1)  # [B, 3D]

        # logits for three NLI labels
        logits = self.classifier(x)
        return logits
