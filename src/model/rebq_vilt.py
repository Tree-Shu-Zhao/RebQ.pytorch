import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.models.vilt.modeling_vilt import (ViltAttention,
                                                    ViltEmbeddings,
                                                    ViltEncoder, ViltLayer,
                                                    ViltPooler,
                                                    ViltPreTrainedModel,
                                                    ViltSelfAttention)
from transformers.utils import logging

logger = logging.get_logger(__name__)


"""Our prompted Vilt"""
class PromptedVilt(ViltPreTrainedModel):
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config

        self.embeddings = ViltEmbeddings(config)
        self.encoder = PromptedViltEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViltPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.text_embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        image_token_type_idx: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        vision_prompts=None,
        vision_prompt_position=None,
        vision_prompt_layers=None, 
        language_prompts=None,
        language_prompt_position=None,
        language_prompt_layers=None, 
        memory=None,
        memory_prompt_position=None,
        memory_prompt_layers=None, 
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.FloatTensor]]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltModel
        >>> from PIL import Image
        >>> import requests

        >>> # prepare image and text
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "hello world"

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        >>> model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

        >>> inputs = processor(image, text, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        text_batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((text_batch_size, seq_length)), device=device)

        if pixel_values is not None and image_embeds is not None:
            raise ValueError("You cannot specify both pixel_values and image_embeds at the same time")
        elif pixel_values is None and image_embeds is None:
            raise ValueError("You have to specify either pixel_values or image_embeds")

        image_batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeds.shape[0]
        if image_batch_size != text_batch_size:
            raise ValueError("The text inputs and image inputs need to have the same batch size")
        if pixel_mask is None:
            pixel_mask = torch.ones((image_batch_size, self.config.image_size, self.config.image_size), device=device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, attention_mask = self.embeddings(
            input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            pixel_mask,
            inputs_embeds,
            image_embeds,
            image_token_type_idx=image_token_type_idx,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            vision_prompts=vision_prompts,
            vision_prompt_position=vision_prompt_position,
            vision_prompt_layers=vision_prompt_layers, 
            language_prompts=language_prompts,
            language_prompt_position=language_prompt_position,
            language_prompt_layers=language_prompt_layers, 
            memory=memory,
            memory_prompt_position=memory_prompt_position,
            memory_prompt_layers=memory_prompt_layers, 
            text_len=input_ids.size()[-1],
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class PromptedViltEncoder(ViltEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([PromptedViltLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True, 
        vision_prompts=None,
        vision_prompt_position=None,
        vision_prompt_layers=None, 
        language_prompts=None,
        language_prompt_position=None,
        language_prompt_layers=None, 
        memory=None,
        memory_prompt_position=None,
        memory_prompt_layers=None,
        text_len=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        hidden_states = hidden_states.detach()

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                batch_size = hidden_states.shape[0]
                device = hidden_states.device
                attention_prompt = False

                # Memory prompts
                if memory is not None and i < memory_prompt_layers:
                    if memory_prompt_position == "input":
                        memory_len = memory[0].shape[1]
                        if i == 0:
                            hidden_states = torch.cat((
                                hidden_states[:, :1, :],
                                memory[i],
                                hidden_states[:, 1:, :]
                            ), dim=1)
                            memory_masks = torch.zeros(batch_size, 1, 1, memory_len, device=device).long()
                            attention_mask = torch.cat((memory_masks, attention_mask), dim=-1)
                        else:
                            hidden_states = torch.cat((
                                hidden_states[:, :1, :],
                                memory[i],
                                hidden_states[:, 1+memory_len:, :]
                            ), dim=1)
                    elif memory_prompt_position == "attention":
                        layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions, memory=memory[i])
                        attention_prompt = True
                
                # Text prompts
                if language_prompts is not None \
                    and language_prompt_position == "input" \
                    and i < language_prompt_layers:
                    assert memory is None
                    language_prompt_len = language_prompts[0].shape[1]
                    if i == 0:
                        hidden_states = torch.cat((
                            hidden_states[:, :1, :],
                            language_prompts[i],
                            hidden_states[:, 1:, :]
                        ), dim=1)
                        language_prompt_masks = torch.zeros(batch_size, 1, 1, language_prompt_len, device=device).long()
                        attention_mask = torch.cat((language_prompt_masks, attention_mask), dim=-1)
                    else:
                        hidden_states = torch.cat((
                            hidden_states[:, :1, :],
                            language_prompts[i],
                            hidden_states[:, 1+language_prompt_len:, :]
                        ), dim=1)

                # Carefully insert vision prompts due to it's depend on language prompts
                if vision_prompts is not None \
                    and vision_prompt_position == "input" \
                    and i < vision_prompt_layers:
                    assert memory is None
                    vision_prompt_len = vision_prompts[0].shape[1]
                    language_prompt_len = 0 if language_prompts is None or language_prompt_position == "attention" else language_prompts[0].shape[1]
                    if i == 0:
                        hidden_states = torch.cat((
                            hidden_states[:, :text_len+language_prompt_len+1, :],
                            vision_prompts[i],
                            hidden_states[:, text_len+language_prompt_len+1:, :]
                        ), dim=1)
                        vision_prompt_masks = torch.zeros(batch_size, 1, 1, vision_prompt_len, device=device).long()
                        attention_mask = torch.cat((vision_prompt_masks, attention_mask), dim=-1)
                    else:
                        hidden_states = torch.cat((
                            hidden_states[:, :text_len+language_prompt_len+1, :],
                            vision_prompts[i],
                            hidden_states[:, text_len+language_prompt_len+vision_prompt_len+1:, :]
                        ), dim=1)

                # We need to firstly handle all input prompts, and then feed them to layers
                
                # Text prompts and Vision prompts, attention
                if language_prompts is not None \
                    and language_prompt_position == "attention" \
                    and i < language_prompt_layers \
                    and vision_prompts is not None \
                    and vision_prompt_position == "attention" \
                    and i < vision_prompt_layers:
                    assert memory is None
                    assert attention_prompt is False
                    layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions, language_prompts=language_prompts[i], vision_prompts=vision_prompts[i], text_len=text_len)
                    attention_prompt = True
                
                # Text prompt, attention; vision prompts, input
                if language_prompts is not None \
                    and language_prompt_position == "attention" \
                    and i < language_prompt_layers \
                    and i < vision_prompt_layers \
                    and (vision_prompts is None \
                        or vision_prompt_position == "input"):
                    assert memory is None
                    assert attention_prompt is False
                    layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions, language_prompts=language_prompts[i], text_len=text_len)
                    attention_prompt = True
                
                # Vision is not None, Language is None
                if vision_prompts is not None and language_prompts is None:
                    if i < vision_prompt_layers and vision_prompt_position == "attention":
                        layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions, vision_prompts=vision_prompts[i], text_len=text_len)
                        attention_prompt = True
                
                # Text prompt, input; vision prompts, attention
                elif vision_prompts is not None \
                    and vision_prompt_position == "attention" \
                    and i < vision_prompt_layers \
                    and i < language_prompt_layers \
                    and (language_prompts is None \
                        or language_prompt_position == "input"):
                    assert memory is None
                    assert attention_prompt is False
                    layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions, vision_prompts=vision_prompts[i], text_len=text_len)
                    attention_prompt = True

                if attention_prompt == False:
                    layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class PromptedViltLayer(ViltLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = PromptedViltAttention(config)
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, 
        memory=None, language_prompts=None, vision_prompts=None, text_len=None,
    ):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViLT, layernorm is applied before self-attention
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            memory=memory, 
            language_prompts=language_prompts, 
            vision_prompts=vision_prompts,
            text_len=text_len,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states.to(attention_output.device)

        # in ViLT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs

class PromptedViltAttention(ViltAttention):
    def __init__(self, config):
        super().__init__(config)
        self.attention = PromptedViltSelfAttention(config)
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, 
        memory=None, language_prompts=None, vision_prompts=None, text_len=None,
    ):
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions, 
            memory=memory, 
            language_prompts=language_prompts, 
            vision_prompts=vision_prompts,
            text_len=text_len,
        )

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class PromptedViltSelfAttention(ViltSelfAttention):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, 
        memory=None, language_prompts=None, vision_prompts=None, text_len=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        if memory is not None:
            pk, pv = memory
            pk = pk.reshape(len(hidden_states), -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
            pv = pv.reshape(len(hidden_states), -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
            key_layer = torch.cat((pk, key_layer), dim=2)
            value_layer = torch.cat((pv, value_layer), dim=2)
        if language_prompts is not None and vision_prompts is None:
            lpk, lpv = language_prompts
            lpk = lpk.reshape(len(hidden_states), -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
            lpv = lpv.reshape(len(hidden_states), -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
            key_layer = torch.cat((
                key_layer[:, :, :1, :],
                lpk,
                key_layer[:, :, 1:, :],
            ), dim=2)
            value_layer = torch.cat((
                value_layer[:, :, :1, :],
                lpv,
                value_layer[:, :, 1:, :],
            ), dim=2)
        if language_prompts is not None and vision_prompts is not None:
            lpk, lpv = language_prompts
            lpk = lpk.reshape(len(hidden_states), -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
            lpv = lpv.reshape(len(hidden_states), -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
            vpk, vpv = vision_prompts
            vpk = vpk.reshape(len(hidden_states), -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
            vpv = vpv.reshape(len(hidden_states), -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
            key_layer = torch.cat((
                key_layer[:, :, :1, :],
                lpk,
                key_layer[:, :, 1:text_len+1, :],
                vpk,
                key_layer[:, :, text_len+1:, :]
            ), dim=2)
            value_layer = torch.cat((
                value_layer[:, :, :1, :],
                lpv,
                value_layer[:, :, 1:text_len+1, :],
                vpv,
                value_layer[:, :, text_len+1:, :]
            ), dim=2)
        if language_prompts is None and vision_prompts is not None:
            vpk, vpv = vision_prompts
            vpk = vpk.reshape(len(hidden_states), -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
            vpv = vpv.reshape(len(hidden_states), -1, self.num_attention_heads, self.attention_head_size).permute(0, 2, 1, 3)
            key_layer = torch.cat((
                key_layer[:, :, :text_len+1, :],
                vpk,
                key_layer[:, :, text_len+1:, :],
            ), dim=2)
            value_layer = torch.cat((
                value_layer[:, :, :text_len+1, :],
                vpv,
                value_layer[:, :, text_len+1:, :],
            ), dim=2)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            if memory is not None:
                attention_mask = torch.cat((torch.zeros_like(attention_mask[:, :, :, :pk.shape[2]]), attention_mask), dim=-1)
            if language_prompts is not None and vision_prompts is None:
                attention_mask = torch.cat((
                    attention_mask[:, :, :, :1],
                    torch.zeros_like(attention_mask[:, :, :, :lpk.shape[2]]),
                    attention_mask[:, :, :, 1:],
                ), dim=-1)
            if language_prompts is not None and vision_prompts is not None:
                attention_mask = torch.cat((
                    attention_mask[:, :, :, :1],
                    torch.zeros_like(attention_mask[:, :, :, :lpk.shape[2]]),
                    attention_mask[:, :, :, 1:text_len+1],
                    torch.zeros_like(attention_mask[:, :, :, :vpk.shape[2]]),
                    attention_mask[:, :, :, text_len+1:],
                ), dim=-1)
            if language_prompts is None and vision_prompts is not None:
                attention_mask = torch.cat((
                    attention_mask[:, :, :, :text_len+1],
                    torch.zeros_like(attention_mask[:, :, :, :vpk.shape[2]]),
                    attention_mask[:, :, :, text_len+1:],
                ), dim=-1)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
