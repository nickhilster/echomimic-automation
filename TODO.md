# TODO

## Next step (blocking full automation)

1. Run `scripts/echomimic_setup.ps1` on the GPU machine.
2. Open the cloned repo (`%USERPROFILE%\EchoMimicV3`) and find the current
   Flash inference entrypoint — as of the last check it was one of:
   - `bash run_flash.sh`
   - `python app_mm.py` (Gradio UI, quantized, fits 12GB)
3. Find the config file it reads image/audio/output paths from (a YAML/JSON
   under `configs/` — every EchoMimic version uses this pattern instead of
   plain CLI flags).
4. Paste that config file's contents back into the conversation with Claude.

Once that's done, Claude will write `scripts/generate_video.ps1` — a wrapper
that takes `-Image`, `-Audio`, `-Output` and patches the config + runs
inference in one command. That closes the loop on full automation.

## Later / optional

- [ ] Decide if the accelerated/quantized path (`app_mm.py`) or the plain
      Flash script is the better default — quantized trades some quality for
      more VRAM headroom.
- [ ] Confirm actual total weight download size (not stated precisely in the
      upstream README at time of writing).
- [ ] If native Windows install breaks on a compiled dependency
      (xformers/flash-attn), fall back to WSL2 + the same conda steps.
