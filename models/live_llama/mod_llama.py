import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
# from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
from torch.nn import functional as F
from einops import rearrange
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import logger
from transformers.utils import ModelOutput
from transformers.models.llama.modeling_llama import LlamaAttention,LlamaMLP,LlamaRMSNorm,LlamaFlashAttention2,LlamaSdpaAttention
import math
import torch.utils.checkpoint
import types
import matplotlib.pyplot as plt
# import seaborn as sns

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}

class MoDLlamaConfig(LlamaConfig):
    model_type = "mod_llama"
    def __init__(self,
                 mod_enable=True,
                 mod_mode='sparse',
                 mod_layers_idx=None,
                 capacity_factor=0.5,
                 router_aux_loss_coef=0.01,
                 **kwargs):
        super().__init__(**kwargs)
        self.mod = dict(
                mod_enable=mod_enable,
                mod_mode=mod_mode,
                mod_layers_idx=mod_layers_idx,
                capacity_factor=capacity_factor,
                router_aux_loss_coef=router_aux_loss_coef,
            )

@dataclass
class MoDCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mod_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    mod_loss_list: Optional[Tuple[torch.FloatTensor]] = None
    route_index : Optional[torch.FloatTensor] = None

@dataclass
class MoDBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    mod_loss_list: Optional[Tuple[torch.FloatTensor]] = None

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        v_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class MoDLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, router: nn.Linear ):       
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.router = router
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        v_mask: torch.Tensor = None, 
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        batch_size, seq_len, dim = hidden_states.shape
        # frame_token_len = 10 # naive solution,35 is the len of instruction tokens, 1024 is img tokens(resolution)
        # question_token_len = seq_len - frame_token_len
        router_logits = self.router(self.input_layernorm(hidden_states))
        route_probabilities = F.softmax(router_logits, dim=-1)[:, :, 1]  # align with MOE

        frame_token_len = v_mask.sum()
        v_mask = 1 - v_mask

        probabilities = route_probabilities + v_mask
        capacity_factor = 1 - ((frame_token_len * 0.5 ) / seq_len)
        top_k = int(math.ceil(seq_len * capacity_factor))
        token_weights, token_index = torch.topk(probabilities, top_k, dim=-1)
        
        selected_tokens, index = torch.sort(token_index, dim=1) # both are [bs, 223(topk of tokens)]
        r_weights = torch.gather(route_probabilities, dim=1, index=selected_tokens)
        indices_expanded = selected_tokens.unsqueeze(-1).expand(-1, -1, dim)
        selected_hidden_states = torch.gather(input=hidden_states, dim=1, index=indices_expanded)
        if attention_mask is not None:
            new_attention_mask = torch.gather(attention_mask, 1, selected_tokens)
        else:
            new_attention_mask = attention_mask
        new_position_ids = torch.arange(0,top_k).unsqueeze(0).to(selected_hidden_states.device)
        residual = selected_hidden_states
        selected_hidden_states = self.input_layernorm(selected_hidden_states)
        
        selected_hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=selected_hidden_states,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        selected_hidden_states = residual + selected_hidden_states
        residual = selected_hidden_states
        selected_hidden_states = self.post_attention_layernorm(selected_hidden_states)
        selected_hidden_states = self.mlp(selected_hidden_states) * r_weights.unsqueeze(-1) # now router's weights is in gradient flow
        selected_hidden_states = (residual + selected_hidden_states)
        hidden_states = hidden_states.scatter(dim=1, index=indices_expanded, src=selected_hidden_states) # if hidden_states now is [bs,topk,dim]
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        router_targets = torch.zeros_like(route_probabilities)
        for i in range(batch_size):
            router_targets[i, token_index[i]] = 1
        aux_loss = F.cross_entropy(router_logits.view(-1, 2), router_targets.view(-1).long())
        outputs += (aux_loss,)
        return outputs

class MoDLlamaModel(LlamaModel):
    config_class = MoDLlamaConfig
    
    def __init__(self, config: LlamaConfig):
        super(MoDLlamaModel, self).__init__(config)
    
    def initialize_mod_modules(self, mod_mode, mod_layers_idx, capacity_factor, router_aux_loss_coef):
        self.config.mod['mod_mode'] = mod_mode
        self.config.mod['mod_layers_idx'] = mod_layers_idx
        self.config.mod['capacity_factor'] = capacity_factor # topk = Int(squence_len * capacity_factor)
        self.config.mod['router_aux_loss_coef'] = self.router_aux_loss_coef = router_aux_loss_coef
        num_layers = self.config.num_hidden_layers
        mod_layers_idx_ = mod_layers_idx
        if mod_layers_idx is not None:
            # model_args.mod_mode = 'custom'
            assert len(mod_layers_idx) <= num_layers
            assert max(mod_layers_idx) < num_layers
            assert min(mod_layers_idx) >= 0
        else:
            if mod_mode == "first_half":
                mod_layers_idx_ = list(range(0, num_layers // 2))
            elif mod_mode == "second_half":
                mod_layers_idx_ = list(range(num_layers // 2, num_layers-1))
            elif mod_mode == "sparse":
                mod_layers_idx_ = list(range(num_layers))[::2]
            elif mod_mode == "dense":
                mod_layers_idx_ = list(range(num_layers))
            elif mod_mode == "first_last_dense":
                mod_layers_idx_ = list(range(1, num_layers ))
            elif mod_mode == "last_two_thirds":
                mod_layers_idx_ = list(range(int(num_layers // 3)+2, num_layers))
            elif mod_mode == "first_five_dense":
                mod_layers_idx_ = list(range(5, num_layers - 1))
            elif mod_mode == "arank_mod":
                # Select 1st, 2nd, 3rd, and last layer
                mod_layers_idx_ = list(range(num_layers))
                # Remove the first three layers and the last one
                mod_layers_idx_ = [i for i in mod_layers_idx_ if i not in {0, 2, 4, 6, 8, 10}]
            else:
                raise NotImplementedError(
                    f'Only support ["first_half", "second_half", "sparse", "dense", "first_last_dense","last_two_thirds","first_five_dense","arank_mod"], but found {mod_mode}')
        self.config.mod['mod_layers_idx'] = mod_layers_idx_
        router = nn.Linear(self.config.hidden_size, 2, bias=False)
        for i in range(num_layers):
            self.layers[i] = LlamaDecoderLayer(self.config, i)
        for idx in mod_layers_idx_: # only modify the selected layers
            self.layers[idx] = MoDLlamaDecoderLayer(self.config, idx, router)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: torch.LongTensor = None,
        output_mod_loss: Optional[bool] = True,
        labels: Optional[torch.Tensor] = None,
        capacity_factor : Optional[float] = 0.5,
        v_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, MoDBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        return_legacy_cache = False
        if (use_cache and not isinstance(past_key_values, Cache) and not self.training):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds
        
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        all_mod_loss = [] if output_mod_loss else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    v_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    v_mask=v_mask,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if output_mod_loss:
                if self.training:
                    if len(layer_outputs) >= 2:
                        all_mod_loss.extend([layer_outputs[-1]])
                    else:
                        all_mod_loss.append(torch.tensor(0, device=hidden_states.device, dtype=hidden_states.dtype))
                else:
                    pass

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        # next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        

        return MoDBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            mod_loss_list=all_mod_loss,
        )

class MoDLlamaForCausalLM(LlamaForCausalLM):
    config_class = MoDLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MoDLlamaModel(config)
        self.model.initialize_mod_modules("sparse", None, 0.5, 0.01)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
        self.all_samples_route_probs = []

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        v_mask: torch.Tensor = None
    ) -> Union[Tuple, MoDCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print(self.model)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            v_mask = v_mask,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        mod_loss, mod_losses = None, []
        if len(outputs[-1]) > 0:
            mod_loss_list = outputs[-1]
            for mod_loss in mod_loss_list:
                if mod_loss is not None:
                    mod_losses.append(mod_loss)
            mod_loss = sum(mod_losses) * self.model.router_aux_loss_coef
            if labels is not None:
                loss = loss + mod_loss
        if not return_dict:
            output = (logits,) + outputs[1:]
            output = (mod_loss,) + output if mod_loss is not None else output
            return (loss,) + output if loss is not None else output

        return MoDCausalLMOutputWithPast(
            loss=loss,
            mod_loss = mod_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mod_loss_list=outputs.mod_loss_list,
            route_index = self.all_samples_route_probs
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs