# Deep Learning for Music Analysis and Generation — HW3

Text-to-music with Stable Audio Open:

1. CLAP retrieval of MusicCaps clips for 100 test captions.
2. CFG implementation (`cfg.py`).
3. Generation and evaluation with FAD, KL, CLAP and MQAScore.
4. Improved generation, using APG, a guidance interval, composable guidance, prompt engineering and a negative prompt.

The submitted samples use **CFG, w = 10, seed 0, 100 steps**.

## Setup

```bash
conda create -n deepmir_hw3 python=3.11 -y
conda activate deepmir_hw3
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`stabilityai/stable-audio-open-1.0` is gated on Hugging Face. Accept its license on the model page, then log in:

```bash
huggingface-cli login
```

Generation uses about 9 GB of GPU memory in fp16. On a smaller GPU, add `--offload` to any generation command to enable model CPU offload. This is slower but fits in 8 GB.

## Inference

### Reproduce the submitted samples

This writes `0.wav` … `99.wav` for the captions in `test.csv`:

```bash
python generate_plus.py --out_dir outputs/generated_samples --scale 10 --seed 0
```

- Each clip is 10 s at 44.1 kHz, stereo.
- Clips that already exist are skipped, so an interrupted run can be resumed.
- Add `--limit 5` for a quick test on the first five captions.

### Single prompt

```bash
python generate.py --prompt "jazz, electric guitar, piano, relaxing" --scale 10 --out out.wav
python generate.py --prompt "jazz, electric guitar, piano, relaxing" --check   # compare cfg.py with diffusers' built-in CFG
```

`--check` prints the maximum absolute difference from `StableAudioPipeline`. With this `cfg.py` it is `0.0`.

### Other guidance methods

All of these use `generate_plus.py --out_dir <dir>` plus the options below. They share `--steps` (default 100), `--seconds` (default 10) and `--seed` (default 0).

| Method | Extra options |
| --- | --- |
| CFG (baseline) | `--scale 7` |
| APG | `--guidance apg --scale 10` (or `15`) |
| Guidance interval | `--scale 10 --interval 0 50` (guidance only when σ ≤ 50) |
| Composable guidance | `--scale 7 --concepts concepts.csv --concept_scale 3` |
| Prompt engineering | `--scale 7 --csv prompts_rewritten.csv --col prompt` |
| Negative prompt | `--scale 7 --negative "low quality, noisy, distorted, muffled, amateur recording, silence, vocals, speech"` |
| Fewer steps | `--scale 10 --steps 50` |

## Retrieval, evaluation and analysis (optional)

These scripts expect the data folders to sit **next to** this repository. The datasets are not included.

```
hw3/
├── <this repo>/
├── audio_mp3/               # MusicCaps clips, <ytid>.mp3
├── musiccaps-public.csv     # MusicCaps metadata (Hugging Face google/MusicCaps), used by analyze_extra.py
├── av-benchmark/            # https://github.com/hkchengrex/av-benchmark at f351b9a, with av-benchmark.patch applied
├── retrieval/  eval/  gen/  report/   # created by the scripts
```

| Step | Command |
| --- | --- |
| CLAP retrieval | `python retrieve.py --audio_dir ../audio_mp3 --out_dir ../retrieval` |
| Build the reference set | `python prep_gt.py` (writes `../eval/gt/{id}.wav` and `../eval/text.csv`) |
| FAD / KL / CLAP | `python run_eval.py ../gen/cfg10 [more dirs]` (writes `../eval/results.csv`) |
| MQAScore | `python mqascore.py ../eval/gt ../gen/cfg10 [--shuffle]` |
| MQAScore template probe | `python mqa_probe.py` |
| Tables and figures | `python analyze.py && python analyze_extra.py` (writes `../report/`) |

Notes on the retrieval and evaluation tools:

- **CLAP checkpoint:** retrieval and the CLAP score both use `music_audioset_epoch_15_esc_90.14.pt`, which is downloaded automatically.
- **av-benchmark:** set it up with:

  ```bash
  git clone https://github.com/hkchengrex/av-benchmark ../av-benchmark
  cd ../av-benchmark
  git checkout f351b9a
  git apply ../<this repo>/av-benchmark.patch
  pip install -e .
  ```

  The patch changes the following:
  - It uses the music CLAP checkpoint.
  - It loads video models only when needed.
  - It loads audio with soundfile.
  - It passes `trust_repo` to torch.hub.
  - It updates the msclap tokenizer call.
- **MQAScore:** loads `Qwen/Qwen2-Audio-7B-Instruct` in 4-bit through bitsandbytes.
- **Batch runs:** `run_queue*.ps1` are the PowerShell queues used to run all experiments one after another.

## Files

| File | Description |
| --- | --- |
| `cfg.py` | CFG: `uncond + w · (cond − uncond)` |
| `generate.py` | Stable Audio Open sampling loop with the CFG step replaced by `cfg.py` |
| `gen_all.py` | Batch generation over a caption CSV with `generate.py` |
| `generate_plus.py` | Batch generation with CFG, APG, guidance interval, composable guidance and negative prompt |
| `retrieve.py` | LAION-CLAP text-to-audio retrieval over MusicCaps |
| `prep_gt.py` | Builds the paired reference set from the retrieval results |
| `run_eval.py` | Runs av-benchmark: FD-VGG/PANN/PaSST, KL-PANNs/PaSST and LAION-CLAP |
| `mqascore.py` | MQAScore: per-concept yes/no confidence from Qwen2-Audio |
| `mqa_probe.py` | Compares question templates for MQAScore (matched vs mismatched AUC) |
| `analyze.py`, `analyze_extra.py` | Tables, figures, bootstrap CIs, CLAP–MQAScore correlation and retrieval failure analysis |
| `test.csv` | The 100 test captions |
| `concepts.csv` | Genre, instrument and mood concepts per caption, used by MQAScore and composable guidance |
| `prompts_rewritten.csv` | Tag-style rewritten prompts for prompt engineering |
| `av-benchmark.patch` | Local fixes to av-benchmark |
