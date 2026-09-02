"""
Idempotent patch for upstream EchoMimicV3 `infer_flash.py`.

Upstream parses `--GPU_memory_mode` but never applies it: it calls
`pipeline.to(device)` unconditionally, which loads the whole model stack
(~18 GB: umt5-xxl text encoder + CLIP + transformer + VAE) onto the GPU and
OOMs anything below ~24 GB VRAM.

`WanFunInpaintAudioPipeline` already declares
`model_cpu_offload_seq = "text_encoder->clip_image_encoder->transformer->vae"`,
so it is built for diffusers' offload helpers - this patch just wires
`--GPU_memory_mode` to them:

    sequential_cpu_offload  -> pipeline.enable_sequential_cpu_offload()   (fits ~8 GB, slowest)
    model_cpu_offload       -> pipeline.enable_model_cpu_offload()        (fits ~12 GB)
    normal / anything else  -> pipeline.to(device)                       (needs ~24 GB)

Run:  python scripts/patch_infer_flash.py [REPO_DIR]
Safe to run repeatedly. Keeps a one-time backup at infer_flash.py.orig.
"""
import os
import sys

MARKER = "_emv3_place_pipeline"

HELPER = '''
def _emv3_place_pipeline(pipeline, mode, device):
    """Patched in by echomimic-automation: honour --GPU_memory_mode."""
    m = (mode or "").lower()
    if m in ("sequential_cpu_offload", "sequential"):
        print(f"[emv3] enable_sequential_cpu_offload (mode={m})", flush=True)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif m in ("model_cpu_offload", "model_cpu_offload_and_qfloat8"):
        print(f"[emv3] enable_model_cpu_offload (mode={m})", flush=True)
        pipeline.enable_model_cpu_offload(device=device)
    else:
        print(f"[emv3] pipeline.to({device}) (mode={m or 'normal'})", flush=True)
        pipeline.to(device=device)


'''

# (old, new) string replacements. Each old must occur exactly once.
REPLACEMENTS = [
    # first placement call -> helper
    (
        "\n    pipeline.to(device=device)\n\n    coefficients = get_teacache_coefficients(model_name)",
        "\n    _emv3_place_pipeline(pipeline, GPU_memory_mode, device)\n\n    coefficients = get_teacache_coefficients(model_name)",
    ),
    # redundant second placement call -> drop it
    (
        "    generator = torch.Generator(device=device).manual_seed(seed)\n\n    pipeline.to(device=device)\n",
        "    generator = torch.Generator(device=device).manual_seed(seed)\n",
    ),
]


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/EchoMimicV3")
    path = os.path.join(repo, "infer_flash.py")
    if not os.path.isfile(path):
        sys.exit(f"not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()

    if MARKER in src:
        print("infer_flash.py already patched - nothing to do.")
        return

    orig = src

    # insert helper right before `def main(`
    anchor = "\ndef main("
    if anchor not in src:
        sys.exit("could not find `def main(` in infer_flash.py - upstream layout changed, patch aborted.")
    src = src.replace(anchor, "\n" + HELPER + "def main(", 1)

    for old, new in REPLACEMENTS:
        if src.count(old) != 1:
            sys.exit(
                f"expected exactly one occurrence of a patch anchor, found {src.count(old)}.\n"
                f"Upstream infer_flash.py changed - patch aborted, file untouched.\n"
                f"Anchor:\n{old!r}"
            )
        src = src.replace(old, new, 1)

    backup = path + ".orig"
    if not os.path.exists(backup):
        with open(backup, "w", encoding="utf-8") as fh:
            fh.write(orig)
        print(f"backup written: {backup}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    print("infer_flash.py patched: --GPU_memory_mode is now honoured.")


if __name__ == "__main__":
    main()
