import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


def _minmax_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


class AnomalyAnyCLIPDetector(nn.Module):
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        image_size: int = 224,
        selected_layers: Optional[Sequence[int]] = None,
        prompt_templates: Optional[Sequence[str]] = None,
        use_shared_adapter: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.image_size = image_size
        self.debug = debug
        self.prompt_templates = list(prompt_templates) if prompt_templates else ["a photo of a {} {}."]

        if selected_layers is None:
            selected_layers = [3, 6, 9, 12] if "base" in clip_model_name else [6, 12, 18, 24]
        self.selected_layers = list(selected_layers)

        for p in self.clip.parameters():
            p.requires_grad = False
        assert not any(p.requires_grad for p in self.clip.parameters())

        vision_hidden = self.clip.config.vision_config.hidden_size
        text_hidden = self.clip.config.projection_dim

        self.shared_adapter = use_shared_adapter
        if use_shared_adapter:
            self.adapter = nn.Linear(vision_hidden, text_hidden)
        else:
            self.adapters = nn.ModuleList([nn.Linear(vision_hidden, text_hidden) for _ in self.selected_layers])

        trainable = [p for p in self.parameters() if p.requires_grad]
        assert len(trainable) > 0 and all(p.requires_grad for p in trainable)

        self.memory_banks: Dict[int, torch.Tensor] = {}

    @property
    def trainable_parameters(self):
        if self.shared_adapter:
            return self.adapter.parameters()
        return self.adapters.parameters()

    def build_text_features(self, class_name: str, device: torch.device) -> torch.Tensor:
        normal = [t.format("normal", class_name) for t in self.prompt_templates]
        abnormal = [t.format("anomalous", class_name) for t in self.prompt_templates]
        all_prompts = normal + abnormal
        inputs = self.processor(text=all_prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_feats = self.clip.get_text_features(**inputs)
            text_feats = F.normalize(text_feats, dim=-1)
        n = len(self.prompt_templates)
        f_text = torch.stack([text_feats[:n].mean(dim=0), text_feats[n:].mean(dim=0)], dim=0)
        return F.normalize(f_text, dim=-1)

    def _extract_vision(self, images: torch.Tensor):
        out = self.clip.vision_model(pixel_values=images, output_hidden_states=True)
        pooled = self.clip.visual_projection(out.pooler_output)
        f_image = F.normalize(pooled, dim=-1)

        patch_tokens = []
        for layer_idx in self.selected_layers:
            hs = out.hidden_states[layer_idx]
            tokens = hs[:, 1:, :]
            b, n, c = tokens.shape
            s = int(math.sqrt(n))
            assert s * s == n, f"Patch token count {n} is not square"
            patch_tokens.append((tokens, s, s))
        return f_image, patch_tokens

    def _adapt_patch(self, tokens: torch.Tensor, layer_i: int):
        if self.shared_adapter:
            x = self.adapter(tokens)
        else:
            x = self.adapters[layer_i](tokens)
        return F.normalize(x, dim=-1)

    @torch.no_grad()
    def rebuild_memory_bank(self, normal_loader, device: torch.device):
        self.eval()
        per_layer = {i: [] for i in range(len(self.selected_layers))}
        for images, _, labels, _ in normal_loader:
            images = images.to(device)
            labels = labels.to(device)
            images = images[labels == 0]
            if images.numel() == 0:
                continue
            _, patches = self._extract_vision(images)
            for i, (tok, _, _) in enumerate(patches):
                adapted = self._adapt_patch(tok, i)
                per_layer[i].append(adapted.reshape(-1, adapted.shape[-1]))

        self.memory_banks = {}
        for i, feats in per_layer.items():
            assert len(feats) > 0, f"Empty memory bank for layer {i}"
            self.memory_banks[i] = torch.cat(feats, dim=0).detach()

    def forward(self, images: torch.Tensor, f_text: torch.Tensor):
        b, _, h, w = images.shape
        f_image, patches = self._extract_vision(images)
        s_vl = torch.softmax(f_image @ f_text.t(), dim=-1)
        s_vl_abn = s_vl[:, 1]

        m_vl = torch.zeros((b, 1, h, w), device=images.device)
        m_vv = torch.zeros((b, 1, h, w), device=images.device)

        for i, (tok, ph, pw) in enumerate(patches):
            adapted = self._adapt_patch(tok, i)
            logits = adapted @ f_text.t()
            probs = torch.softmax(logits, dim=-1)[..., 1].reshape(b, 1, ph, pw)
            probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)
            m_vl = m_vl + probs

            bank = self.memory_banks.get(i)
            assert bank is not None and bank.numel() > 0, f"Memory bank layer {i} is empty"
            sim = adapted @ bank.t()
            max_sim = sim.max(dim=-1).values.reshape(b, 1, ph, pw)
            vv = F.interpolate(1.0 - max_sim, size=(h, w), mode="bilinear", align_corners=False)
            m_vv = m_vv + vv

            if self.debug:
                print(f"[DEBUG] layer={self.selected_layers[i]} F_patch={tok.shape} F_hat_patch={adapted.shape}")

        m_vl = _minmax_norm(m_vl)
        m_vv = _minmax_norm(m_vv)
        s_vv = m_vv.flatten(1).max(dim=-1).values

        final_pixel_map = _minmax_norm(m_vl + m_vv)
        image_score = s_vl_abn + s_vv

        if self.debug:
            print(f"[DEBUG] F_image={f_image.shape} F_text={f_text.shape} M_VL={m_vl.shape} M_VV={m_vv.shape}")

        return {
            "image_score": image_score,
            "pixel_map": final_pixel_map,
            "S_VL": s_vl,
            "M_VL": m_vl,
            "M_VV": m_vv,
            "S_VV": s_vv,
        }
