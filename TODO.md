# TODO

## Done
- [x] Environment + weights setup scripted (`scripts/echomimic_setup.ps1`),
      including the pinned dependency set and the wav2vec safetensors conversion.
- [x] Confirmed the Flash inference interface against upstream source.
- [x] `scripts/patch_infer_flash.py` — wires the ignored `--GPU_memory_mode`
      to the pipeline's cpu-offload helpers (for `-Driver flash`).
- [x] `scripts/infer_lowram.py` — load-encode-free driver that fits a 16 GB-RAM
      box.
- [x] `scripts/generate_video.ps1` — `-Image`/`-Audio`/`-Output` wrapper,
      defaults to the low-RAM driver.
- [x] **End-to-end verified** on RTX 4070 (12 GB) + 16 GB RAM: demo 01 + 02
      produce coherent lip-synced video (560x1024, 81f @ 25fps, 8 steps).
- [x] Real (non-demo) inputs verified: a photo + a random slice of an audiobook
      MP3, incl. a cropped head-and-shoulders frame to avoid hand/finger artifacts.
- [x] **Chunked generation** for long clips (`infer_lowram.py`): overlapping
      N-frame chunks conditioned on the previous chunk's tail, cross-faded.
      Verified at 15 s (5 chunks, ~8 min).

## Next / optional
- [ ] Longer clips (30-60 s+) and motion-quality tuning per chunk (`-Steps`).
- [ ] Occasional flaky native crash (`0xC0000409`) during startup imports
      (TF/numba/torch); retry has worked every time so far. Watch for a pattern.
- [ ] Benchmark `-MemMode model_cpu_offload` vs `sequential_cpu_offload` for
      speed vs VRAM headroom on this box.
- [ ] Tune the preset (steps, CFG scales, TeaCache threshold) on real output.
- [ ] If `-Driver flash` is ever needed: raise the pagefile (elevated shell +
      reboot) — infer_flash.py needs ~24 GB RAM.
- [ ] Upstream `torch.cuda.amp.autocast` deprecation warnings are noise now but
      will break on a future torch; pin torch or patch `src/` if upgrading.
