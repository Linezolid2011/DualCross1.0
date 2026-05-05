from typing import Union

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, LlamaConfig, LlamaModel, PreTrainedModel


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    activation: nn.Module = nn.ReLU,
) -> nn.Sequential:
    layers = []
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if n_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())
        layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def get_activation_class(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    elif name == "leakyrelu":
        return nn.LeakyReLU
    elif name == "elu":
        return nn.ELU
    elif name == "selu":
        return nn.SELU
    elif name == "gelu":
        return nn.GELU
    else:
        raise ValueError(f"Unsupported activation function: {name}")


def get_loss_fn(loss: Union[str, nn.Module]) -> nn.Module:
    if isinstance(loss, nn.Module):
        return loss
    loss = loss.lower()
    if loss == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss}")


def get_transformer_backbone(key, kwargs) -> PreTrainedModel:
    kwargs = dict(kwargs or {})
    if key == "GPT2":
        config = GPT2Config(**kwargs)
        model = GPT2BidirectionalModel(config)
        model.wpe.weight.requires_grad = False
        model.wte.weight.requires_grad = False
        model.wpe.weight.zero_()
        model.wte.weight.zero_()
        model_dim = config.n_embd
    elif key == "llama":
        bidirectional_attention = bool(kwargs.pop("bidirectional_attention", False))
        config = LlamaConfig(**kwargs)
        if bidirectional_attention:
            model = LlamaBidirectionalModel(config)
        else:
            model = LlamaModel(config)
        model_dim = config.hidden_size
        model.embed_tokens.weight.requires_grad = False
        model.embed_tokens.weight.zero_()
    else:
        raise ValueError(f"Unknown backbone key {key}")
    return model, model_dim


class NoRoPE(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor):
        batch_size, seq_len, _ = hidden_states.shape
        cos = hidden_states.new_ones(batch_size, seq_len, self.head_dim)
        sin = hidden_states.new_zeros(batch_size, seq_len, self.head_dim)
        return cos, sin


class LlamaBidirectionalModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.rotary_emb = NoRoPE(head_dim=config.head_dim)
        self.config.is_causal = False
        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.is_causal = False

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False):
        return None

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                cache_position=None, **flash_attn_kwargs):
        flash_attn_kwargs["is_causal"] = False
        if attention_mask is None:
            B, S = None, None
            if inputs_embeds is not None:
                B, S = inputs_embeds.size(0), inputs_embeds.size(1)
            if B and S:
                attention_mask = torch.ones((B, 1, S, S), dtype=torch.float, device=inputs_embeds.device)
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            cache_position=cache_position, **flash_attn_kwargs,
        )


class GPT2BidirectionalModel(GPT2Model):
    def __init__(self, config: GPT2Config):
        config.is_decoder = False
        super().__init__(config)
        for block in self.h:
            block.attn.is_causal = False

        def _no_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
            return None
        self._update_causal_mask = _no_causal_mask.__get__(self, GPT2Model)

    def forward(self, input_ids=None, past_key_values=None, cache_position=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                encoder_hidden_states=None, encoder_attention_mask=None, use_cache=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        if attention_mask is not None:
            B, S = attention_mask.size()
            expanded = attention_mask.unsqueeze(1).unsqueeze(2).expand(B, 1, S, S)
            neg_inf = torch.finfo(self.dtype).min
            float_mask = (1.0 - expanded.to(self.dtype)) * neg_inf
        return super().forward(
            input_ids=input_ids, past_key_values=past_key_values, cache_position=cache_position,
            attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids,
            head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, **kwargs,
        )
