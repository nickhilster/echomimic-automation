"""
Low-RAM driver for EchoMimicV3 Flash inference.

Upstream `infer_flash.py` loads text-encoder (11.4 GB) + CLIP (4.7 GB) +
transformer (3 GB) + flash weights (3.7 GB) + VAE all into system RAM before
offloading to GPU -> ~23 GB resident, which OOMs a 16 GB box.

This driver keeps peak RAM to roughly one big module at a time:

  1. load tokenizer + text-encoder  -> encode prompt / negative prompt -> free it
  2. load CLIP image encoder        -> encode the reference image      -> free it
  3. load base transformer + apply flash-pro weights + load VAE
  4. build the pipeline with lightweight stand-ins for the two freed encoders,
     monkeypatch `encode_prompt` / `check_inputs`, run the diffusion loop.

CLI args are a subset of upstream `infer_flash.py` (same names), so the same
wrapper can call either script.
"""
import argparse
import gc
import math
import os
import sys

# This script lives outside the EchoMimicV3 repo, so make `src.*` importable.
_REPO = os.environ.get("ECHOMIMIC_REPO") or os.getcwd()
if os.path.isdir(os.path.join(_REPO, "src")) and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import librosa
import numpy as np
import torch
from einops import rearrange
from moviepy import AudioFileClip, VideoFileClip
from omegaconf import OmegaConf
from PIL import Image
import torchvision.transforms.functional as TF
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor

from src.wan_vae import AutoencoderKLWan
from src.wan_image_encoder import CLIPModel
from src.wan_text_encoder import WanT5EncoderModel
from src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel as WanTransformer
from src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline
from src.utils import get_image_to_video_latent3, save_videos_grid
from src.fm_solvers import FlowDPMSolverMultistepScheduler
from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
from src.cache_utils import get_teacache_coefficients
from src.wav2vec2 import Wav2Vec2Model
from diffusers import FlowMatchEulerDiscreteScheduler

DEFAULT_NEG = ("Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. "
               "Bad fingers. Unclear and blurry hands. Unclear gestures, broken hands, "
               "fused fingers. 手指融合，")


def parse_args():
    p = argparse.ArgumentParser(description="EchoMimicV3 Flash - low RAM driver")
    p.add_argument("--config_path", type=str, default="config/config.yaml")
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--transformer_path", type=str, required=True)
    p.add_argument("--wav2vec_model_dir", type=str, required=True)
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--audio_path", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--save_path", type=str, default="outputs")
    p.add_argument("--sampler_name", type=str, default="Flow_Unipc",
                   choices=["Flow", "Flow_Unipc", "Flow_DPM++"])
    p.add_argument("--video_length", type=int, default=81)
    p.add_argument("--partial_video_length", type=int, default=81,
                   help="frames per chunk; clips longer than this are generated in overlapping chunks")
    p.add_argument("--overlap_video_length", type=int, default=8,
                   help="frames blended between consecutive chunks")
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--audio_guidance_scale", type=float, default=3.0)
    p.add_argument("--audio_scale", type=float, default=1.0)
    p.add_argument("--neg_scale", type=float, default=1.0)
    p.add_argument("--neg_steps", type=int, default=0)
    p.add_argument("--num_inference_steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--enable_teacache", action="store_true", default=False)
    p.add_argument("--teacache_threshold", type=float, default=0.1)
    p.add_argument("--num_skip_start_steps", type=int, default=5)
    p.add_argument("--teacache_offload", action="store_true", default=False)
    p.add_argument("--enable_riflex", action="store_true", default=False)
    p.add_argument("--riflex_k", type=int, default=6)
    p.add_argument("--mem_mode", type=str, default="model_cpu_offload",
                   choices=["model_cpu_offload", "sequential_cpu_offload", "cuda"])
    p.add_argument("--weight_dtype", type=str, default="bfloat16",
                   choices=["float16", "bfloat16"])
    p.add_argument("--sample_size", type=int, nargs=2, default=[768, 768])
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--negative_prompt", type=str, default=DEFAULT_NEG)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--cfg_skip_ratio", type=float, default=0.0)
    # accepted for wrapper compatibility, unused here
    p.add_argument("--ulysses_degree", type=int, default=1)
    p.add_argument("--ring_degree", type=int, default=1)
    p.add_argument("--GPU_memory_mode", type=str, default=None)
    p.add_argument("--ckpt_idx", type=int, default=50000)
    p.add_argument("--add_prompt", type=str, default="")
    return p.parse_args()


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_sample_size(pil_img, sample_size):
    w, h = pil_img.size
    ori_a = w * h
    default_a = sample_size[0] * sample_size[1]
    if default_a < ori_a:
        ratio_a = math.sqrt(ori_a / sample_size[0] / sample_size[1])
        w = w / ratio_a // 16 * 16
        h = h / ratio_a // 16 * 16
    else:
        w = w // 16 * 16
        h = h // 16 * 16
    return [int(h), int(w)]


def loudness_norm(audio_array, sr=16000, lufs=-23):
    import pyloudnorm as pyln
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    return pyln.normalize.loudness(audio_array, loudness, lufs)


def get_audio_embed(mel_input, feat_extractor, audio_encoder, video_length, sr=16000, device="cpu"):
    audio_feature = np.squeeze(feat_extractor(mel_input, sampling_rate=sr).input_values)
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device).unsqueeze(0)
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d").cpu().detach()
    return audio_emb


class _StubBase(torch.nn.Module):
    """Param-less stand-in for a freed encoder. Satisfies diffusers' pipeline
    introspection (`.dtype`, `.device`, `.to()`, `.parameters()`) without holding
    any weights."""
    def __init__(self, dtype, device):
        super().__init__()
        self._dtype = dtype
        self._device = torch.device(device) if not isinstance(device, torch.device) else device

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def to(self, *a, **k):
        for x in list(a) + list(k.values()):
            if isinstance(x, (str, torch.device)):
                try:
                    self._device = torch.device(x)
                except (RuntimeError, TypeError):
                    pass
            elif isinstance(x, torch.dtype):
                self._dtype = x
        return self

    def parameters(self, recurse=True):
        return iter(())

    def buffers(self, recurse=True):
        return iter(())


class _TextEncStub(_StubBase):
    def forward(self, *a, **k):
        raise RuntimeError("text encoder was freed; prompt embeds are precomputed")


class _ClipStub(_StubBase):
    """Returns the precomputed CLIP context regardless of input."""
    def __init__(self, ctx, dtype, device):
        super().__init__(dtype, device)
        self._ctx = ctx

    def __call__(self, *a, **k):
        return self._ctx

    def forward(self, *a, **k):
        return self._ctx


def main():
    args = parse_args()
    weight_dtype = torch.bfloat16 if args.weight_dtype == "bfloat16" else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = OmegaConf.load(args.config_path)
    os.makedirs(args.save_path, exist_ok=True)

    image_name = os.path.basename(args.image_path).split(".")[0]
    output_video_path = os.path.join(args.save_path, f"{image_name}_output.mp4")
    if os.path.exists(output_video_path):
        print(f"skip: {output_video_path} exists")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")))

    # ---- 1. text encoder: load -> encode -> free -----------------------------
    print("[lowram] loading text encoder ...", flush=True)
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_name, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    class _Duck:
        pass
    duck = _Duck()
    duck.tokenizer = tokenizer
    duck.text_encoder = text_encoder
    duck._execution_device = torch.device("cpu")

    print("[lowram] encoding prompt ...", flush=True)
    with torch.no_grad():
        prompt_embeds = WanFunInpaintAudioPipeline._get_t5_prompt_embeds(
            duck, prompt=args.prompt, num_videos_per_prompt=1,
            max_sequence_length=512, device=torch.device("cpu"), dtype=weight_dtype)
        negative_prompt_embeds = WanFunInpaintAudioPipeline._get_t5_prompt_embeds(
            duck, prompt=args.negative_prompt or "", num_videos_per_prompt=1,
            max_sequence_length=512, device=torch.device("cpu"), dtype=weight_dtype)
    prompt_embeds = [e.to(device=device, dtype=weight_dtype) for e in prompt_embeds]
    negative_prompt_embeds = [e.to(device=device, dtype=weight_dtype) for e in negative_prompt_embeds]

    del text_encoder, duck
    free()
    print("[lowram] text encoder freed", flush=True)

    # ---- 2. audio features -------------------------------------------------
    audio_encoder = Wav2Vec2Model.from_pretrained(args.wav2vec_model_dir, local_files_only=True).to("cpu")
    audio_encoder.feature_extractor._freeze_parameters()
    feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_model_dir, local_files_only=True)

    audio_clip = AudioFileClip(args.audio_path)
    tcr = 4  # vae temporal compression ratio (from config)
    video_length_actual = min(int(audio_clip.duration * args.fps), args.video_length)
    video_length_actual = (int((video_length_actual - 1) // tcr * tcr) + 1) if video_length_actual != 1 else 1

    mel_input, sr = librosa.load(args.audio_path, sr=16000)
    mel_input = loudness_norm(mel_input, sr)
    mel_input = mel_input[:int(video_length_actual / 25 * sr)]
    audio_feat = get_audio_embed(mel_input, feat_extractor, audio_encoder, video_length_actual)
    del audio_encoder
    free()

    audio_embeds = audio_feat.to(device=device, dtype=weight_dtype)
    indices = (torch.arange(2 * 2 + 1) - 2) * 1
    center_indices = torch.arange(0, video_length_actual, 1).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=audio_embeds.shape[0] - 1)
    audio_embeds = audio_embeds[center_indices].unsqueeze(0).to(device=device)
    print(f"[lowram] audio embeds {tuple(audio_embeds.shape)}", flush=True)

    # ---- image geometry + clip image ------------------------------------------
    ref_image = Image.open(args.image_path).convert("RGB")
    sample_h, sample_w = get_sample_size(ref_image, args.sample_size)

    def _round_tcr(n):
        return (int((n - 1) // tcr * tcr) + 1) if n != 1 else 1

    partial = _round_tcr(min(args.partial_video_length, video_length_actual))
    overlap = min(args.overlap_video_length, max(partial - 1, 1))

    # CLIP context comes from the ORIGINAL reference every chunk (identity anchor).
    _iv, _ivm, clip_image = get_image_to_video_latent3(
        ref_image, None, video_length=partial, sample_size=[sample_h, sample_w])
    del _iv, _ivm

    # ---- 3. CLIP: load -> encode -> free ------------------------------------
    print("[lowram] loading CLIP image encoder ...", flush=True)
    clip_encoder = CLIPModel.from_pretrained(
        os.path.join(args.model_name, config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"))
    ).to(weight_dtype).eval().to(device)
    with torch.no_grad():
        clip_t = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, weight_dtype)
        clip_context = clip_encoder([clip_t[:, None, :, :]])
    clip_context = clip_context.detach()
    del clip_encoder, clip_t
    free()
    print("[lowram] CLIP freed", flush=True)

    # ---- 4. transformer + VAE ---------------------------------------------
    print("[lowram] loading transformer ...", flush=True)
    transformer = WanTransformer.from_pretrained(
        os.path.join(args.model_name, config["transformer_additional_kwargs"].get("transformer_subpath", "transformer")),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=False,
        torch_dtype=weight_dtype,
    )
    from safetensors.torch import load_file
    print(f"[lowram] applying flash weights: {args.transformer_path}", flush=True)
    state_dict = load_file(args.transformer_path)
    state_dict = state_dict.get("state_dict", state_dict)
    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"[lowram] transformer load: missing={len(m)} unexpected={len(u)}", flush=True)
    del state_dict
    free()

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    # ---- scheduler ----------------------------------------------------------
    sched_cls = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[args.sampler_name]
    sk = OmegaConf.to_container(config["scheduler_kwargs"])
    if args.sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        sk["shift"] = 1
    from src.utils import filter_kwargs
    scheduler = sched_cls(**filter_kwargs(sched_cls, sk))

    # ---- 5. pipeline with stubs ------------------------------------------
    pipeline = WanFunInpaintAudioPipeline(
        tokenizer=tokenizer,
        text_encoder=_TextEncStub(weight_dtype, device),
        vae=vae,
        transformer=transformer,
        clip_image_encoder=_ClipStub(clip_context, weight_dtype, device),
        scheduler=scheduler,
    )
    pipeline.encode_prompt = lambda *a, **k: (prompt_embeds, negative_prompt_embeds)
    pipeline.check_inputs = lambda *a, **k: None
    pipeline.model_cpu_offload_seq = "transformer->vae"

    if args.mem_mode == "sequential_cpu_offload":
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.mem_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device)

    if args.enable_teacache:
        coeff = get_teacache_coefficients(args.model_name)
        if coeff is not None:
            pipeline.transformer.enable_teacache(
                coeff, args.num_inference_steps, args.teacache_threshold,
                num_skip_start_steps=args.num_skip_start_steps, offload=args.teacache_offload)

    if args.enable_riflex:
        pipeline.transformer.enable_riflex(k=args.riflex_k, L_test=(partial - 1) // tcr + 1)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # ---- 6. chunked diffusion --------------------------------------------
    # The model's window is ~one chunk; longer clips are generated in
    # overlapping chunks, each conditioned on the tail frames of the last,
    # then linearly cross-faded over `overlap` frames (mirrors app_mm.py).
    n_chunks = 1 + max(0, -(-(video_length_actual - partial) // max(partial - overlap, 1)))
    print(f"[lowram] chunked gen: {video_length_actual} frames | {partial}/chunk | "
          f"{overlap} overlap | ~{n_chunks} chunk(s)", flush=True)

    init_frames = 0
    last_frames = partial
    new_sample = None
    rolling = ref_image          # PIL for chunk 0; list[PIL] thereafter
    cur_partial = partial
    idx = 0
    while init_frames < video_length_actual:
        if last_frames >= video_length_actual:
            cur_partial = _round_tcr(video_length_actual - init_frames)
            if cur_partial <= 0:
                break
        idx += 1
        print(f"[lowram] chunk {idx}: frames {init_frames}-{init_frames + cur_partial}", flush=True)

        iv, ivm, _ = get_image_to_video_latent3(
            rolling, None, video_length=cur_partial, sample_size=[sample_h, sample_w])
        a_slice = audio_embeds[:, init_frames:init_frames + cur_partial]

        with torch.no_grad():
            sample = pipeline(
                args.prompt,
                num_frames=cur_partial,
                negative_prompt=args.negative_prompt,
                audio_embeds=a_slice,
                audio_scale=args.audio_scale,
                ip_mask=None,
                use_un_ip_mask=False,
                height=sample_h,
                width=sample_w,
                generator=generator,
                neg_scale=args.neg_scale,
                neg_steps=args.neg_steps,
                guidance_scale=args.guidance_scale,
                audio_guidance_scale=args.audio_guidance_scale,
                num_inference_steps=args.num_inference_steps,
                video=iv,
                mask_video=ivm,
                clip_image=clip_image,
                cfg_skip_ratio=args.cfg_skip_ratio,
                shift=args.shift,
            ).videos
        sample = sample.float().cpu()
        free()

        if init_frames != 0:
            k = min(overlap, sample.shape[2], new_sample.shape[2])
            mix = torch.tensor([i / overlap for i in range(k)], dtype=sample.dtype).view(1, 1, k, 1, 1)
            new_sample[:, :, -k:] = new_sample[:, :, -k:] * (1 - mix) + sample[:, :, :k] * mix
            new_sample = torch.cat([new_sample, sample[:, :, k:]], dim=2)
        else:
            new_sample = sample

        if last_frames >= video_length_actual:
            break

        rolling = [
            Image.fromarray(
                (new_sample[0, :, i].permute(1, 2, 0) * 255).clamp(0, 255).byte().numpy())
            for i in range(-overlap, 0)
        ]
        init_frames += cur_partial - overlap
        last_frames = init_frames + cur_partial

    final_frames = new_sample.shape[2]
    print(f"[lowram] stitched {final_frames} frames", flush=True)

    tmp_video_path = os.path.join(args.save_path, f"{image_name}_tmp.mp4")
    save_videos_grid(new_sample[:, :, :final_frames], tmp_video_path, fps=args.fps)

    video_clip = VideoFileClip(tmp_video_path)
    audio_clip = audio_clip.subclipped(0, final_frames / args.fps)
    video_clip = video_clip.with_audio(audio_clip)
    video_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", threads=2)
    os.remove(tmp_video_path)
    print(f"Saved output to: {output_video_path}", flush=True)


if __name__ == "__main__":
    main()
