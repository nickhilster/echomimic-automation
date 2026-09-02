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

## Next / optional
- [ ] Try real (non-demo) image+audio inputs and longer audio clips.
- [ ] Benchmark `-MemMode model_cpu_offload` vs `sequential_cpu_offload` for
      speed vs VRAM headroom on this box.
- [ ] Tune the preset (steps, CFG scales, TeaCache threshold) on real output.
- [ ] If `-Driver flash` is ever needed: raise the pagefile (elevated shell +
      reboot) — infer_flash.py needs ~24 GB RAM.
- [ ] Upstream `torch.cuda.amp.autocast` deprecation warnings are noise now but
      will break on a future torch; pin torch or patch `src/` if upgrading.
