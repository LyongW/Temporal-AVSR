import sys,logging
import contextlib
from argparse import Namespace

import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import FairseqEncoder, FairseqEncoderModel, register_model, BaseFairseqModel, FairseqEncoderDecoderModel
from typing import Any, Optional
from fairseq import utils
import math

from fairseq.dataclass import FairseqDataclass
from omegaconf import II, MISSING

from fairseq.modules import LayerNorm
from pathlib import Path
from transformers import  AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch.distributed as dist
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache
import torch.nn.functional as F
from avhubert.hubert_asr import AVHubertAsrConfig, HubertEncoderWrapper
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperModel
from .sub_model.modules import WhisperEncoderWrapper, Projector, Multimodal_Attention, Speech_Rate_Predictor
from .sub_model.Qformer import BertConfig, BertLMHeadModel
from torch.nn.utils.rnn import pad_sequence
from .sub_model.ctc import CTC
from .sub_model.modeling_llama import LlamaForCausalLM, LlamaConfig, LlamaDecoderLayer


logger = logging.getLogger(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2] # torch.Size([1, 606, 16, 32])
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings for temporal modeling
    Args:
        tensor: [batch, seq_len, num_heads, head_dim] or [1, seq_len, num_heads, head_dim]
        freqs: [seq_len, head_dim//2] - must match tensor's seq_len
    """
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    # Expand to match tensor dimensions: [1, seq_len, 1, head_dim]
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    if tensor.shape[1] != cos.shape[1]:
        print(tensor.shape, cos.shape, sin.shape)  # torch.Size([1, 606, 16, 64]) torch.Size([1, 606, 1, 64]) torch.Size([1, 606, 1, 64])
    output = (tensor * cos) + (rotate_half(tensor) * sin) # torch.Size([1, 606, 16, 64]) * torch.Size([1, 606, 1, 64])
    output = output.to(orig_dtype)
    return output # torch.Size([1, 606, 16, 64])

class TemporalRotaryEmbedding(nn.Module):
    """Rotary position embeddings for temporal dimension"""
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

class TemporalMultiFrameAttention(nn.Module):
    """Temporal attention module for processing temporal relationships between video frames"""
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (total_seq_length, dim) - flattened batch
            cu_seqlens: cumulative sequence lengths for batch samples
            rotary_pos_emb: (total_seq_length, head_dim//2) - per-frame position embeddings
        """
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # q, k, v: [seq_length, num_heads, head_dim]
        
        # Apply rotary position embeddings
        if rotary_pos_emb is not None:
            # rotary_pos_emb should have shape [seq_length, head_dim//2]
            q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
            k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        # Create attention mask based on cumulative sequence lengths
        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        ) #torch.Size([1, 606, 606])
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = 0

        q = q.transpose(0, 1) # torch.Size([16, 606, 64])
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output

class VisualFeaturePooler(nn.Module):
    """用于将visual features池化到指定长度"""
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        
    def forward(self, visual_feat: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Args:
            visual_feat: [batch, seq_len, hidden_dim] 例如 [6, 101, 1024]
            target_length: 目标长度,例如压缩后的 len_queries
        Returns:
            pooled_feat: [batch, target_length, hidden_dim]
        """
        batch_size, seq_len, h_dim = visual_feat.shape
        
        # 将序列维度转换为2D进行池化: [B, D, T] -> [B, D, H, W]
        # 这里将时间序列看作1D,转换为接近正方形的2D
        hw = int(seq_len ** 0.5)
        
        # 如果seq_len不是完全平方数,进行padding
        if hw * hw != seq_len:
            hw = int(seq_len ** 0.5) + 1
            padded_len = hw * hw
            padding_size = padded_len - seq_len
            # 在时间维度末尾padding
            visual_feat = F.pad(visual_feat, (0, 0, 0, padding_size), mode='constant', value=0)
        
        # Reshape: [B, T, D] -> [B, D, H, W]
        shaped_visual_feat = visual_feat.transpose(1, 2).view(batch_size, h_dim, hw, hw)
        
        # 计算目标尺寸
        target_hw = int(target_length ** 0.5)
        if target_hw * target_hw != target_length:
            target_hw = int(target_length ** 0.5) + 1
        
        # 自适应平均池化
        sampler = nn.AdaptiveAvgPool2d((target_hw, target_hw))
        pooled_visual_feat = sampler(shaped_visual_feat)  # [B, D, target_hw, target_hw]
        
        # Reshape back: [B, D, H, W] -> [B, target_length, D]
        reshaped_visual_feat = pooled_visual_feat.view(batch_size, h_dim, -1).transpose(1, 2)
        
        # 如果池化后长度大于target_length,进行截断
        if reshaped_visual_feat.size(1) > target_length:
            reshaped_visual_feat = reshaped_visual_feat[:, :target_length, :]
        # 如果池化后长度小于target_length,进行padding
        elif reshaped_visual_feat.size(1) < target_length:
            padding_size = target_length - reshaped_visual_feat.size(1)
            reshaped_visual_feat = F.pad(reshaped_visual_feat, (0, 0, 0, padding_size), mode='constant', value=0)
        
        return reshaped_visual_feat

class LlamaDecoderLayerWithVisual(LlamaDecoderLayer):
    """扩展的LlamaDecoderLayer,支持视觉特征注入"""
    def __init__(self, config, layer_idx, inject_visual=False):
        super().__init__(config, layer_idx)
        self.inject_visual = inject_visual
        if inject_visual:
            # 视觉特征维度与LLM隐藏层维度应该匹配
            visual_hidden_size = getattr(config, 'visual_hidden_size', config.hidden_size)
            # 如果维度不匹配，需要投影层
            if visual_hidden_size != config.hidden_size:
                self.visual_projection = nn.Linear(visual_hidden_size, config.hidden_size)
            else:
                self.visual_projection = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        # 先调用原始LlamaDecoderLayer的forward
        outputs = super().forward(
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
        
        hidden_states = outputs[0]
        
        return outputs


class LlamaForCausalLMWithVisual(LlamaForCausalLM):
    """扩展的LlamaForCausalLM,支持视觉特征注入"""
    def __init__(self, config):
        super().__init__(config)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):
        """扩展forward方法,支持传入视觉特征
        Args:
            visual_pos_masks: [batch, seq_len] bool tensor
            deepstack_visual_embeds: List of [num_visual_tokens, hidden_size], 
                                     长度为需要注入的层数
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 准备输入
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        # 调用model的forward，会遍历所有decoder layers
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-num_logits_to_keep, None) if isinstance(num_logits_to_keep, int) else num_logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@dataclass
class TemporalAVSRConfig(AVHubertAsrConfig):
    llm_path: str = field(
        default='meta-llama/Llama-3.2-3B'
    )
    target_modules: str = field(
        default='q_proj.v_proj.k_proj.o_proj'
    )
    whisper_embed_dim: int = field(
        default=1024, metadata={"help": "whisper embedding dimension"}
    )
    avhubert_embed_dim: int = field(
        default=1024, metadata={"help": "avhubert embedding dimension"}
    )
    llama_embed_dim: int = field(
        default=3072, metadata={"help": "llama embedding dimension"}
    )
    lora_rank: int = field(
        default=16, metadata={"help": "lora_rank"}
    )
    lora_alpha: int = field(
        default=32, metadata={"help": "lora_alpha"}
    )
    modality_fuse: str = field(
        default='concat', metadata={'help': 'fusing two modalities: concat, add, cross-att'}
    )
    ### Speech Q-Former Config ###
    use_qformer: bool = field(
        default=True
    )
    window_level: bool = field(
        default=False
    )
    queries_per_sec: int = field(
        default=4, metadata={"help": "queries_per_sec"}
    )
    qformer_layers: int = field(
        default=2, metadata={"help": "number of qformer layers"}
    )
    qformer_dim: int = field(
        default=1024, metadata={"help": "qformer dim"}
    )
    whisper_path: str = field(default="openai/whisper-medium.en")
    qformer_path: str = field(default="bert-large-uncased")
    sr_predictor_path: str = field(default="pretrained_models/sr_predictor/checkpoint.pt")

    ### Speech Rate Predictor Config ###
    use_sr_predictor: bool = field(
        default=False
    )
    sr_predictor_layers: int = field(
        default=2, metadata={"help": "number of sr predictor layers"}
    )

     ### Temporal Attention Config ###
    use_temporal_attention: bool = field(
        default=True, metadata={"help": "whether to use temporal attention"}
    )
    temporal_num_heads: int = field(
        default=16, metadata={"help": "number of heads for temporal attention"}
    )
    temporal_num_layers: int = field(
        default=2, metadata={"help": "number of temporal attention layers"}
    )

    ### CTC Config ###
    use_ctc: bool = field(
        default=True, metadata={"help": "whether to use CTC loss"}
    )
    ctc_weight: float = field(
        default=0.1, metadata={"help": "weight for CTC loss"}
    )
    vocab_size: int = field(
        default=128256, metadata={"help": "vocabulary size for CTC"}
    )

    visual_hidden_size: int = field(
        default=1024, metadata={"help": "visual features hidden size"}
    )
    visual_inject_layers: list = field(
        default_factory=lambda: [14, 16, 18], metadata={"help": "layers to inject visual features"}
    )
    use_quantization: bool = field(
        default=True, metadata={"help": "whether to use quantization"}
    )
              
@register_model("Temporal-AVSR", dataclass=TemporalAVSRConfig)
class TemporalAVSR(BaseFairseqModel):    
    def __init__(self, avhubert, whisper, llm, tokenizer, cfg):
        super().__init__() 
        self.cfg = cfg
        self.avhubert = avhubert
        self.whisper = whisper
        self.llama = llm   

        self.tokenizer = tokenizer
        
        for param in self.avhubert.parameters():
            param.requires_grad = False
            
        for param in self.whisper.parameters():
            param.requires_grad = False
    
        self.modality_fuse = cfg.modality_fuse
        if self.modality_fuse == 'concat':
            self.embed = cfg.whisper_embed_dim + cfg.avhubert_embed_dim
        elif self.modality_fuse == 'add':
            self.embed = cfg.whisper_embed_dim
        elif self.modality_fuse == 'cross-att':
            self.multimodal_attention_layer = Multimodal_Attention(embed_dim=cfg.whisper_embed_dim, num_heads=8)
            self.embed = cfg.whisper_embed_dim

        if cfg.use_temporal_attention:
            self.temporal_rotary_emb = TemporalRotaryEmbedding(cfg.avhubert_embed_dim // cfg.temporal_num_heads)
            self.temporal_attention_layers = nn.ModuleList([
                TemporalMultiFrameAttention(dim=cfg.avhubert_embed_dim, num_heads=cfg.temporal_num_heads)
                for _ in range(cfg.temporal_num_layers)
            ])
            self.temporal_norm = nn.LayerNorm(cfg.avhubert_embed_dim)
                
        #### Qformer ####
        if cfg.use_qformer:
            if cfg.window_level:
                cfg.max_queries = 1
            self.afeat_1d_conv = nn.Conv1d(in_channels=cfg.whisper_embed_dim, out_channels=cfg.whisper_embed_dim, kernel_size=2, stride=2, padding=0) # 50Hz -> 25Hz
            if cfg.use_sr_predictor:
                max_queries = int(cfg.queries_per_sec * 20 * 2)
            else:
                max_queries = int(cfg.queries_per_sec * 20)
            
            qformer_config = BertConfig.from_pretrained(cfg.qformer_path)
            qformer_config.num_hidden_layers = cfg.qformer_layers
            qformer_config.encoder_width = self.embed
            qformer_config.hidden_size = cfg.qformer_dim 
            qformer_config.add_cross_attention = True
            qformer_config.cross_attention_freq = 1
            qformer_config.query_length = max_queries
            self.Qformer = BertLMHeadModel(config=qformer_config)
            self.query_tokens = nn.Parameter(
                torch.zeros(1, max_queries, qformer_config.hidden_size)
            )
            self.query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)
            

            if cfg.use_sr_predictor:
                max_queries = int(cfg.queries_per_sec * 20 * 2)
                self.sr_predictor = Speech_Rate_Predictor(num_layers=cfg.sr_predictor_layers)
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                sr_ckpt_path = cfg.sr_predictor_path
                sr_state = torch.load(sr_ckpt_path, map_location='cpu')['model']
                sr_state_ = {}
                for k, v in sr_state.items():
                    sr_state_[k[13:]] = v
                self.sr_predictor.load_state_dict(sr_state_)
                for param in self.sr_predictor.parameters():
                    param.requires_grad = False

            self.avfeat_to_llm = Projector(input_dim=qformer_config.hidden_size,
                                        hidden_dim=math.floor((qformer_config.hidden_size + self.cfg.llama_embed_dim)/2),
                                        output_dim=self.cfg.llama_embed_dim) 
        else:
            self.afeat_1d_conv = nn.Conv1d(in_channels=cfg.whisper_embed_dim, out_channels=cfg.whisper_embed_dim, kernel_size=4, stride=4, padding=0) # 50Hz -> 12.5Hz
            self.vfeat_1d_conv = nn.Conv1d(in_channels=cfg.whisper_embed_dim, out_channels=cfg.whisper_embed_dim, kernel_size=2, stride=2, padding=0) # 25Hz -> 12.5Hz
            self.avfeat_to_llm = Projector(input_dim=self.embed,
                                        hidden_dim=math.floor((self.embed + self.cfg.llama_embed_dim)/2),
                                        output_dim=self.cfg.llama_embed_dim) 
            
            
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.freeze_params = [n for n,p in self.named_parameters() if p.requires_grad == False]
        
        # CTC module
        if cfg.use_ctc:
            self.ctc = CTC(
                odim=cfg.vocab_size,
                eprojs=cfg.avhubert_embed_dim,  # 1024
                dropout_rate=0.1,
                reduce=True
            )
            # 确保 CTC 模块使用 float32
            self.ctc = self.ctc.float()
            self.ctc_weight = cfg.ctc_weight
        else:
            self.ctc = None
            self.ctc_weight = 0.0

        if cfg.visual_inject_layers:
            self.visual_pooler = VisualFeaturePooler(hidden_size=cfg.visual_hidden_size)
        
        self.layer_results = nn.LayerNorm(cfg.avhubert_embed_dim)

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        w2v_path = cfg.w2v_path

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data

        task_pretrain = tasks.setup_task(w2v_args.task)
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)

        avhubert = HubertEncoderWrapper(encoder_)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            del state['model']['mask_emb'] 
            avhubert.w2v_model.load_state_dict(state["model"], strict=False)

        avhubert.w2v_model.remove_pretraining_modules()

        whisper_ = WhisperForConditionalGeneration.from_pretrained(cfg.whisper_path).model.encoder
        whisper = WhisperEncoderWrapper(whisper_)

        if cfg.use_quantization:
            bnb_config = BitsAndBytesConfig( 
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # 直接加载量化的预训练模型
            llm_base = LlamaForCausalLMWithVisual.from_pretrained(
                cfg.llm_path,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16
            )
        else:
            llm_base = LlamaForCausalLMWithVisual.from_pretrained(
                cfg.llm_path,
                torch_dtype=torch.bfloat16
            )

        target_modules = cfg.target_modules.split('.')
        config = LoraConfig(
            r=cfg.lora_rank, 
            lora_alpha=cfg.lora_alpha, 
            target_modules=target_modules, 
            lora_dropout=0.05, 
            bias="none", 
            task_type="CAUSAL_LM" 
        )

        # 应用LoRA
        if cfg.use_quantization:
            from peft import prepare_model_for_kbit_training
            llm_base = prepare_model_for_kbit_training(llm_base)

        llm = get_peft_model(llm_base, config)
        llm.print_trainable_parameters()

        tokenizer = AutoTokenizer.from_pretrained(cfg.llm_path)

        return cls(avhubert, whisper, llm.base_model.model, tokenizer, cfg)
    

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
        
    def state_dict(self):
        old_state = super().state_dict()
        state = {k:v for k,v in old_state.items() if k not in self.freeze_params}
        return state
    
    def load_state_dict(self,state,**kwargs):
        super().load_state_dict(state, strict=False)   

    def forward(self, **kwargs):
        # ============================
        # 1. Whisper & AVHubert feature extraction (no grad)
        # ============================
        has_audio = kwargs['source']['audio'] is not None
        with torch.no_grad():
            if has_audio:
            # Whisper encoder: B x T x D
                whisper_enc_out = self.whisper(kwargs['source']) # torch.Size([32, 1500, 1024])
            # Prepare input for AVHubert (audio is None)
            avhubert_source = {'audio': None, 'video': kwargs['source']['video']}
            avhubert_output = self.avhubert(source=avhubert_source, padding_mask=kwargs['padding_mask'])
            # Transpose from (T x B x D) to (B x T x D) torch.Size([32, 1, 29, 88, 88])---->torch.Size([32, 29, 1024])
            avhubert_output['encoder_out'] = avhubert_output['encoder_out'].transpose(0, 1)

        video_lengths = torch.sum(~avhubert_output['padding_mask'], dim=1).tolist()
        max_vid_len = max(video_lengths)
        
        # ============================
        # 2. Speech rate predictor and query length calculation
        # ============================
        
        if has_audio and self.cfg.use_sr_predictor:
            len_queries, resized_len_list = self.query_length_calculation(whisper_enc_out, video_lengths, max_vid_len,has_audio)
        else:
            # len_queries = [max(int(vid_len / 25 * self.cfg.queries_per_sec), self.cfg.queries_per_sec) for vid_len in video_lengths]
            len_queries, resized_len_list = self.query_length_calculation(avhubert_output['encoder_out'], video_lengths, max_vid_len,has_audio)
            # resized_len_list = None
        # ============================
        # 3. Feature processing and modality fusion
        # ============================
        if has_audio:
            whisper_enc_out = self.afeat_1d_conv(whisper_enc_out.transpose(1, 2)).transpose(1, 2) # Process Whisper features with 1D conv: (B x T x D) -> (B x T x D')
        # torch.Size([32, 750, 1024]) 

        if self.cfg.use_qformer:
            padding_mask = (~avhubert_output['padding_mask']).long()
            len_feat = video_lengths 
        else:
            # Without Qformer: downsample the visual feature and corresponding padding mask (e.g., 25Hz -> 12.5Hz)
            padding_mask = avhubert_output['padding_mask'][:, 1::2]
            padding_mask = (~padding_mask).long()
            len_feat = torch.sum(padding_mask, dim=1).tolist()
            avhubert_output['encoder_out'] = self.vfeat_1d_conv(
                avhubert_output['encoder_out'].transpose(1, 2)
            ).transpose(1, 2)

        B, T_v, dim = avhubert_output['encoder_out'].size()

        if self.cfg.use_temporal_attention:
            # Prepare cumulative sequence lengths based on MAX length (for masking)
            cu_seqlens = torch.arange(0, (B + 1) * T_v, T_v, device=avhubert_output['encoder_out'].device, dtype=torch.int32)
            
            # Generate temporal rotary embeddings - pad to max_vid_len for each sample
            temporal_pos_emb_list = []
            for vid_len in video_lengths:
                # Generate position embeddings for actual length
                pos_emb = self.temporal_rotary_emb(vid_len)  # [vid_len, head_dim//2]
                
                # Pad to max_vid_len if necessary
                if vid_len < T_v:
                    # Pad with zeros (or repeat last position)
                    pad_size = T_v - vid_len
                    # Option 1: Zero padding
                    padding = torch.zeros(pad_size, pos_emb.size(1), device=pos_emb.device, dtype=pos_emb.dtype)
                    pos_emb = torch.cat([pos_emb, padding], dim=0)
                    
                    # Option 2 (alternative): Repeat last position
                    # padding = pos_emb[-1:].repeat(pad_size, 1)
                    # pos_emb = torch.cat([pos_emb, padding], dim=0)
                
                temporal_pos_emb_list.append(pos_emb)
            
            # Concatenate all position embeddings: [B * T_v, head_dim//2]
            temporal_pos_emb = torch.cat(temporal_pos_emb_list, dim=0)
            
            # Reshape for temporal attention: (B x T x D) -> (B*T_v, D)
            temporal_input = avhubert_output['encoder_out'].reshape(-1, dim)
            
            # Create proper attention mask using actual video lengths
            # This ensures padded positions don't attend to anything
            attention_mask_valid = torch.zeros(B, T_v, dtype=torch.bool, device=temporal_input.device)
            for i, vid_len in enumerate(video_lengths):
                attention_mask_valid[i, :vid_len] = True
            
            # Apply temporal attention layers
            for temporal_layer in self.temporal_attention_layers:
                residual = temporal_input
                temporal_output = temporal_layer(
                    temporal_input, cu_seqlens=cu_seqlens, rotary_pos_emb=temporal_pos_emb
                )
                
                # Mask out padded positions in the residual connection
                mask_flat = attention_mask_valid.reshape(-1, 1)  # [B*T_v, 1]
                temporal_input = torch.where(mask_flat, residual + temporal_output, residual)
            
            temporal_input = self.temporal_norm(temporal_input)
            # Reshape back: (B*T_v, D) -> (B, T_v, D)
            avhubert_output['encoder_out'] = temporal_input.reshape(B, T_v, dim)

        # CTC Loss Calculation (if enabled)
        loss_ctc = 0.0
        if self.ctc is not None:
            # Prepare CTC labels: remove <BOS> and <EOS> tokens (128000, 128001)
            labels_list = kwargs['target_list']  # List[torch.Tensor]
            ctc_labels = []
            for label in labels_list:
                # Remove BOS and EOS tokens
                label_clean = label[(label != 128000) & (label != 128001)]
                ctc_labels.append(label_clean)
            
            # Pad labels to create batch tensor
            ctc_labels_pad = torch.nn.utils.rnn.pad_sequence(
                ctc_labels, 
                batch_first=True, 
                padding_value=self.ctc.ignore_id
            )
            
            # avhubert_output['encoder_out']: [B, T_v, 1024]
            ctc_input = avhubert_output['encoder_out']  # [B, T_v, 1024]
            ctc_lengths = torch.tensor(video_lengths, device=ctc_input.device, dtype=torch.long)
            
            # Compute CTC loss
            loss_ctc, _ = self.ctc(ctc_input, ctc_lengths, ctc_labels_pad)

        if has_audio:
            whisper_enc_out = whisper_enc_out[:, :T_v, :] # torch.Size([6, 101, 1024])
            # Fuse modalities based on configuration
            if self.modality_fuse == 'concat': # torch.Size([6, 101, 1024])
                av_feat = torch.cat([whisper_enc_out, avhubert_output['encoder_out']], dim=2) # torch.Size([32, 29, 1024]) +++ torch.Size([32, 29, 1024]) ===torch.Size([32, 29, 2048])
            elif self.modality_fuse == 'add':
                av_feat = whisper_enc_out + avhubert_output['encoder_out']
            elif self.modality_fuse == 'cross-att':
                av_feat = self.multimodal_attention_layer(
                    audio_feature=whisper_enc_out,
                    visual_feature=avhubert_output['encoder_out']
                )
            else:
                raise ValueError(f"Unknown modality fusion type: {self.modality_fuse}")
        else:
            # whisper_enc_out = torch.flip(avhubert_output['encoder_out'], dims=[1])  
            # whisper_enc_out = torch.zeros(B, T_v, dim, device=avhubert_output['encoder_out'].device, dtype=avhubert_output['encoder_out'].dtype)
            whisper_enc_out = avhubert_output['encoder_out']
            if self.modality_fuse == 'concat': 
                av_feat = torch.cat([whisper_enc_out, avhubert_output['encoder_out']], dim=2) # torch.Size([32, 29, 1024]) +++ torch.Size([32, 29, 1024]) ===torch.Size([32, 29, 2048])
            elif self.modality_fuse == 'add':
                av_feat = whisper_enc_out + avhubert_output['encoder_out']
            elif self.modality_fuse == 'cross-att':
                av_feat = self.multimodal_attention_layer(
                    audio_feature=whisper_enc_out,
                    visual_feature=avhubert_output['encoder_out']
                )
            else:
                raise ValueError(f"Unknown modality fusion type: {self.modality_fuse}")

        # ============================
        # 4. Prepare inputs for LLM (using Qformer or not)
        # ============================
        instructions = kwargs['source']['instruction']  # List[torch.Tensor] of length B
        labels = kwargs['target_list']                   # List[torch.Tensor] of length B

        if self.cfg.use_qformer:
            query_output = self.compression_using_qformer(len_queries, resized_len_list, len_feat, av_feat)
            # Map Qformer output to LLM embedding space
            query_output = self.avfeat_to_llm(query_output)
            llm_inputs, attention_mask, llm_labels, adjusted_visual_positions = self.prepare_inputs_labels_for_queries(
                instructions, query_output, len_queries, labels
            )
            deepstack_visual_embeds = []
            for layer_inx in [-2, -4]:
                visual_feat_injected = query_output.reshape(-1, query_output.size(-1))
                deepstack_visual_embeds.append(visual_feat_injected)

            visual_pos_masks = self._create_visual_mask(adjusted_visual_positions, llm_inputs.size(1), llm_inputs.device)
        else:
            # Directly map fused AV features to LLM embedding space
            av_feat = self.avfeat_to_llm(av_feat)
            llm_inputs, attention_mask, llm_labels,adjusted_visual_positions = self.prepare_inputs_labels_for_queries(
                instructions, av_feat, len_feat, labels
            )
            deepstack_visual_embeds = []
            for layer_inx in [-2, -4, -6]:
                visual_feat_flat = avhubert_output['layer_results'][layer_inx][0].transpose(0, 1)
                deepstack_visual_embeds.append(visual_feat_flat)
            
            visual_pos_masks = self._create_visual_mask(adjusted_visual_positions, llm_inputs.size(1), llm_inputs.device)
        
        # ============================
        # 5. Forward through the LLM and return outputs
        # ============================

        llm_out = self.llama(
            inputs_embeds=llm_inputs,
            attention_mask=attention_mask,
            labels=llm_labels,
            return_dict=True,
            use_cache=False,
            deepstack_visual_embeds=deepstack_visual_embeds,
            visual_pos_masks=visual_pos_masks,
        )
        # output_hidden_states=True,

        loss_llm = llm_out.loss
        logits = llm_out.logits
        
        # Combine LLM loss and CTC loss
        if self.ctc is not None:
            loss = (1 - self.ctc_weight) * loss_llm + self.ctc_weight * loss_ctc
        else:
            loss = loss_llm
        

        return loss, logits, llm_labels


    @torch.no_grad()
    def generate(self,
                num_beams=5,
                temperature=0.3,
                max_length=100,
                min_length=1,
                **kwargs):

        # --------------------------------------
        # 1. Whisper & AVHubert Feature Extraction
        # --------------------------------------
        has_audio = kwargs['source']['audio'] is not None
        if has_audio:
            whisper_enc_out = self.whisper(kwargs['source'])

        avhubert_source = {'audio': None, 'video': kwargs['source']['video']}
        avhubert_output = self.avhubert(source=avhubert_source, padding_mask=kwargs['padding_mask'])
        # AVHubert encoder output: (T x B x D) -> (B x T x D)
        avhubert_output['encoder_out'] = avhubert_output['encoder_out'].transpose(0, 1)

        video_lengths = torch.sum(~avhubert_output['padding_mask'], dim=1).tolist()
        max_vid_len = max(video_lengths)

        if has_audio and self.cfg.use_sr_predictor:
            len_queries, resized_len_list = self.query_length_calculation(whisper_enc_out, video_lengths, max_vid_len,has_audio)
        else:
            len_queries, resized_len_list = self.query_length_calculation(avhubert_output['encoder_out'], video_lengths, max_vid_len,has_audio)
            
        # --------------------------------------
        # 2. Whisper Feature Processing
        # --------------------------------------
        if has_audio:
            whisper_enc_out = self.afeat_1d_conv(whisper_enc_out.transpose(1, 2)).transpose(1, 2)

        # --------------------------------------
        # 3. AVHubert Feature & Padding Mask Preparation
        # --------------------------------------
        if self.cfg.use_qformer:
            padding_mask = (~avhubert_output['padding_mask']).long()
            len_feat = video_lengths  
        else:
            padding_mask = avhubert_output['padding_mask'][:, 1::2]
            padding_mask = (~padding_mask).long()
            len_feat = torch.sum(padding_mask, dim=1).tolist()
            avhubert_output['encoder_out'] = self.vfeat_1d_conv(
                avhubert_output['encoder_out'].transpose(1, 2)
            ).transpose(1, 2)

        # --------------------------------------
        # 4. Temporal Alignment & Modality Fusion
        # --------------------------------------
        B, T_v, dim = avhubert_output['encoder_out'].size()

        if self.cfg.use_temporal_attention:
            # Use uniform sequence lengths for computation
            cu_seqlens = torch.arange(0, (B + 1) * T_v, T_v, device=avhubert_output['encoder_out'].device, dtype=torch.int32)
            
            # Generate and pad position embeddings
            temporal_pos_emb_list = []
            for vid_len in video_lengths:
                pos_emb = self.temporal_rotary_emb(vid_len)
                
                # Pad to max_vid_len
                if vid_len < T_v:
                    pad_size = T_v - vid_len
                    padding = torch.zeros(pad_size, pos_emb.size(1), device=pos_emb.device, dtype=pos_emb.dtype)
                    pos_emb = torch.cat([pos_emb, padding], dim=0)
                
                temporal_pos_emb_list.append(pos_emb)
            
            temporal_pos_emb = torch.cat(temporal_pos_emb_list, dim=0)  # [B*T_v, head_dim//2]
            
            temporal_input = avhubert_output['encoder_out'].reshape(-1, dim)
            
            # Create attention mask for valid positions
            attention_mask_valid = torch.zeros(B, T_v, dtype=torch.bool, device=temporal_input.device)
            for i, vid_len in enumerate(video_lengths):
                attention_mask_valid[i, :vid_len] = True
            
            for temporal_layer in self.temporal_attention_layers:
                residual = temporal_input
                temporal_output = temporal_layer(
                    temporal_input, cu_seqlens=cu_seqlens, rotary_pos_emb=temporal_pos_emb
                )
                
                mask_flat = attention_mask_valid.reshape(-1, 1)
                temporal_input = torch.where(mask_flat, residual + temporal_output, residual)
            
            temporal_input = self.temporal_norm(temporal_input)
            avhubert_output['encoder_out'] = temporal_input.reshape(B, T_v, dim)

        if has_audio:
            whisper_enc_out = whisper_enc_out[:, :T_v, :]
            if self.modality_fuse == 'concat':
                av_feat = torch.cat([whisper_enc_out, avhubert_output['encoder_out']], dim=2)
            elif self.modality_fuse == 'add':
                av_feat = whisper_enc_out + avhubert_output['encoder_out']
            elif self.modality_fuse == 'cross-att':
                av_feat = self.multimodal_attention_layer(
                    audio_feature=whisper_enc_out,
                    visual_feature=avhubert_output['encoder_out']
                )
            else:
                raise ValueError(f"Unknown modality fusion type: {self.modality_fuse}")
        else:
            # whisper_enc_out = torch.flip(avhubert_output['encoder_out'], dims=[1])
            # whisper_enc_out = torch.zeros(B, T_v, dim, device=avhubert_output['encoder_out'].device, dtype=avhubert_output['encoder_out'].dtype)
            whisper_enc_out = avhubert_output['encoder_out']
            if self.modality_fuse == 'concat': 
                av_feat = torch.cat([whisper_enc_out, avhubert_output['encoder_out']], dim=2) # torch.Size([32, 29, 1024]) +++ torch.Size([32, 29, 1024]) ===torch.Size([32, 29, 2048])
            elif self.modality_fuse == 'add':
                av_feat = whisper_enc_out + avhubert_output['encoder_out']
            elif self.modality_fuse == 'cross-att':
                av_feat = self.multimodal_attention_layer(
                    audio_feature=whisper_enc_out,
                    visual_feature=avhubert_output['encoder_out']
                )
            else:
                raise ValueError(f"Unknown modality fusion type: {self.modality_fuse}")

        # --------------------------------------
        # 5. Prepare inputs for LLM (using Qformer or not)
        # --------------------------------------
        instructions = kwargs['source']['instruction']  # List[torch.Tensor], B
        if self.cfg.use_qformer:
            query_output = self.compression_using_qformer(len_queries, resized_len_list, len_feat, av_feat)
            query_output = self.avfeat_to_llm(query_output)
            llm_inputs, attention_mask, _ ,adjusted_visual_positions= self.prepare_inputs_labels_for_queries(
                instructions, query_output, len_queries
            )
        else:
            # Directly map fused AV features to LLM embedding space
            av_feat = self.avfeat_to_llm(av_feat)
            llm_inputs, attention_mask, _ ,adjusted_visual_positions= self.prepare_inputs_labels_for_queries(
                instructions, av_feat, len_feat
            )

        # 6. LLM Generation
        self.llama.generation_config.pad_token_id = self.tokenizer("<|finetune_right_pad_id|>").input_ids[1]

        outputs = self.llama.generate(
            inputs_embeds=llm_inputs,
            attention_mask=attention_mask,
            num_beams=num_beams,
            temperature=temperature,
            max_new_tokens=max_length,
            min_length=min_length
        )

        return outputs
        
        
    def prepare_inputs_labels_for_queries(self, instructions, queries, len_queries, labels=None):
        llm_input_list = []
        llm_labels_list = []
        lengths = []  
        visual_positions = []

        for i in range(len(instructions)):
            instruction = instructions[i]
            len_query = len_queries[i]
            query = queries[i][:len_query, :]

            inst_emb = self.llama.model.embed_tokens(instruction.unsqueeze(0)).squeeze(0)
            query_start = inst_emb.size(0)
            if labels is not None:
                label = labels[i]
                label_emb = self.llama.model.embed_tokens(label.unsqueeze(0)).squeeze(0)
                combined = torch.cat([inst_emb, query, label_emb], dim=0)
            else:
                combined = torch.cat([inst_emb, query], dim=0)

            query_end = query_start + len_query
            visual_positions.append([query_start, query_end])

            llm_input_list.append(combined)
            lengths.append(combined.size(0)) 

            if labels is not None:
                label_mask = torch.full((combined.size(0),), -100, dtype=instruction.dtype, device=instruction.device)
                offset = inst_emb.size(0) + query.size(0)
                label_mask[offset:] = label
                llm_labels_list.append(label_mask)

        # Determine the maximum sequence length across the batch
        max_seq_len = max(lengths)
        batch_size = len(llm_input_list)
        embedding_dim = llm_input_list[0].size(1)

        # Prepare the pad embedding (using the provided pad token)
        pad_token_id = self.tokenizer("<|finetune_right_pad_id|>").input_ids[1]
        pad_token_tensor = torch.tensor([pad_token_id], device=instruction.device)
        pad_embedding = self.llama.model.embed_tokens(pad_token_tensor).squeeze(0)

        # Initialize the left-padded inputs tensor with the pad embedding.
        # Each sequence will occupy the rightmost positions.
        llm_inputs = pad_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, max_seq_len, embedding_dim).clone()
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=instruction.device)

        adjusted_visual_positions = []
        for i, seq in enumerate(llm_input_list):
            seq_len = seq.size(0)
            # Place the sequence at the right end, leaving pad tokens on the left
            llm_inputs[i, max_seq_len - seq_len:] = seq
            attention_mask[i, max_seq_len - seq_len:] = 1

            padding_offset = max_seq_len - seq_len
            adjusted_start = visual_positions[i][0] + padding_offset
            adjusted_end = visual_positions[i][1] + padding_offset
            adjusted_visual_positions.append([adjusted_start, adjusted_end])

        if labels is not None:
            llm_labels = torch.full((batch_size, max_seq_len), -100, dtype=instruction.dtype, device=instruction.device)
            for i, lab in enumerate(llm_labels_list):
                lab_len = lab.size(0)
                llm_labels[i, max_seq_len - lab_len:] = lab
        else:
            llm_labels = None

        return llm_inputs, attention_mask, llm_labels, adjusted_visual_positions
    
    def query_length_calculation(self, whisper_enc_out, video_lengths, max_vid_len, has_audio):
        if has_audio:
            with torch.no_grad():
                sr_predictions = self.sr_predictor(whisper_enc_out[:,:2*max_vid_len,:][:,::4,:])
            len_queries = []
            resized_len_list = []
            for i, vid_len in enumerate(video_lengths):
                base_queries = vid_len / 25 * self.cfg.queries_per_sec
                factor = sr_predictions[i].item()
                # If predicted speech rate is out of acceptable range, use factor 1.0
                if factor < 1: 
                    factor = 1
                elif factor > 2:
                    factor = 2
                adjusted_queries = int(base_queries * factor)
                query_count = max(adjusted_queries, self.cfg.queries_per_sec)
                len_queries.append(query_count)
                resized_len_list.append(factor*vid_len) # resized av feat
        else:
            len_queries = [max(int(x / 25 * self.cfg.queries_per_sec), self.cfg.queries_per_sec) for x in video_lengths]
            resized_len_list = [vid_len for vid_len in video_lengths]
            max_len_queries = max(len_queries)
            len_queries = [max_len_queries] * len(len_queries)

        return len_queries, resized_len_list

    def compression_using_qformer(self, len_queries, resized_len_list, len_feat, av_feat):
        max_length = max(len_queries) # av_feat:torch.Size([6, 101, 2048])
        B = len(len_queries)
        # Create attention mask for query tokens: (B x max_length)
        query_attn_mask = torch.zeros(B, max_length, dtype=torch.long, device=av_feat.device)
        for i, qlen in enumerate(len_queries):
            query_attn_mask[i, :qlen] = 1

        # Expand and slice query tokens: (B x max_length x token_dim)
        query_tokens = self.query_tokens.expand(B, -1, -1)[:, :max_length, :] # torch.Size([32, 6, 1024])


        resized_av_feats = torch.zeros(B,int(max(resized_len_list)),av_feat.size(2)).to(av_feat.device).to(av_feat.dtype) # torch.Size([32, 58, 2048])
        resized_padding_masks=torch.zeros(B,int(max(resized_len_list))).to(av_feat.device).to(av_feat.dtype)
        # Resize av_feat depend on the factor_list

        for bs,len_feat_bs in enumerate(len_feat): 
            new_av_feat=av_feat[bs][:len_feat_bs].transpose(0, 1).unsqueeze(0) # 1 x D x T
            resized_av_feat = F.interpolate(new_av_feat, size=int(resized_len_list[bs]), mode='linear')
            resized_av_feat=resized_av_feat.squeeze(0).transpose(0,1)
            resized_av_feats[bs,:resized_av_feat.size(0)]=resized_av_feat
            resized_padding_masks[bs,:int(resized_len_list[bs])]=1
            
        av_feat = resized_av_feats # torch.Size([6, 135, 2048])
        padding_mask = resized_padding_masks.long()

        # Run Qformer (using its BERT) with cross attention to AV features
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            attention_mask=query_attn_mask,
            encoder_hidden_states=av_feat,
            encoder_attention_mask=padding_mask,
            return_dict=True
        )['last_hidden_state'] # torch.Size([32, 6, 1024])
    
        return query_output

    def _create_visual_mask(self, visual_positions, seq_len, device):
        """创建视觉特征mask
        Args:
            visual_positions: List[[start, end], ...] 每个样本的视觉特征位置
            seq_len: 序列总长度
            device: tensor设备
        Returns:
            visual_mask: [batch, seq_len] bool tensor
        """
        batch_size = len(visual_positions)
        visual_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        for i, (start, end) in enumerate(visual_positions):
            visual_mask[i, start:end] = True
        
        return visual_mask
        
        
def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


# Compatibility names for checkpoints produced before the project rename.
MMS_LLaMA_Config = TemporalAVSRConfig
MMS_LLaMA = TemporalAVSR
register_model("MMS-LLaMA", dataclass=TemporalAVSRConfig)(TemporalAVSR)
