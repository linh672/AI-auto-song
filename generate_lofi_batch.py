"""
ACE-Step V1.5 — Batch Music Generation Script with tqdm Progress Tracking
Generates 5 rounds of 2-batch music tracks (10 tracks total) for a given description.
"""

import os
import sys
import time
from pathlib import Path
from loguru import logger
from tqdm.auto import tqdm

# Ensure proxy settings do not interfere
for proxy_var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(proxy_var, None)

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.gpu_config import (
    get_gpu_config,
    resolve_lm_backend,
    VRAM_AUTO_OFFLOAD_THRESHOLD_GB,
    is_mps_platform,
)

# Output directory for generated music
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "lofi_chillhop")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# User prompt
MUSIC_PROMPT = (
    "Lo-fi hip hop and chillhop with laid-back swung drums, mellow piano loops, "
    "soft jazzy chord stabs, a round bassline, and subtle head-nod bounce; "
    "intro opens with vinyl crackle and felt-piano fragments, verses keep it sparse "
    "with dusty hats and warm bass, middle section adds brushed percussion and filtered "
    "chord movement, then the outro melts into tape wobble and reversed piano tails, "
    "Intimate close-mic texture, cozy and nostalgic, soft-edged and warmly compressed, "
    "nostalgic, smooth, soft, bassline, jazzy, mellow, hip hop, relaxing"
)

# Configuration for 8GB VRAM (RTX 4060)
TOTAL_ROUNDS = 10  # 10 individual generations
BATCH_SIZE = 1     # Batch size 1 prevents VRAM thrashing / timeout on 8GB GPU
TARGET_DURATION = 75.0  # Target duration in seconds (~1m15s Lo-fi track)


class TqdmProgressTracker:
    """Handles real-time tqdm progress bars for both overall batches and step-by-step phases."""

    def __init__(self, total_rounds: int, batch_size: int):
        self.total_rounds = total_rounds
        self.batch_size = batch_size
        self.current_round = 0
        self.overall_bar = tqdm(
            total=total_rounds * batch_size,
            desc="🎵 Total Tracks Progress",
            unit="track",
            position=0,
            leave=True,
        )
        self.round_bar = None

    def start_round(self, round_idx: int):
        self.current_round = round_idx
        if self.round_bar is not None:
            self.round_bar.close()
        self.round_bar = tqdm(
            total=100,
            desc=f"⏳ Round {round_idx}/{self.total_rounds} [Phase: Initializing]",
            unit="%",
            position=1,
            leave=False,
        )

    def update_step_progress(self, progress_val, desc: str = ""):
        if self.round_bar is not None:
            if isinstance(progress_val, (int, float)):
                pct = int(min(1.0, max(0.0, float(progress_val))) * 100)
                self.round_bar.n = pct
                clean_desc = desc.replace("\n", " ").strip() if desc else "Processing"
                self.round_bar.set_description(f"⏳ Round {self.current_round}/{self.total_rounds} [{clean_desc[:35]}]")
                self.round_bar.refresh()

    def finish_round(self, num_tracks_generated: int):
        if self.round_bar is not None:
            self.round_bar.n = 100
            self.round_bar.refresh()
            self.round_bar.close()
            self.round_bar = None
        self.overall_bar.update(num_tracks_generated)

    def close(self):
        if self.round_bar is not None:
            self.round_bar.close()
        self.overall_bar.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Detect GPU & determine offload settings
    gpu_config = get_gpu_config()
    vram_gb = gpu_config.gpu_memory_gb
    _is_mac = is_mps_platform()
    auto_offload = (not _is_mac) and (0 < vram_gb < VRAM_AUTO_OFFLOAD_THRESHOLD_GB)

    backend = resolve_lm_backend(None, gpu_config)
    logger.info(f"GPU Detected: {vram_gb:.1f} GB VRAM | Offload to CPU: {auto_offload} | LM Backend: {backend}")

    # 2. Initialize DiT model (acestep-v15-turbo)
    logger.info("Initializing DiT model (acestep-v15-turbo)...")
    t0 = time.time()
    dit_handler = AceStepHandler()
    status_msg, success = dit_handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path="acestep-v15-turbo",
        device="auto",
        offload_to_cpu=auto_offload,
        offload_dit_to_cpu=auto_offload,
    )
    if not success:
        logger.error(f"Failed to initialize DiT model: {status_msg}")
        sys.exit(1)
    logger.info(f"DiT model ready in {time.time() - t0:.1f}s")

    # 3. Initialize 5Hz LM model for CoT / audio codes (1.7B model)
    lm_model = "acestep-5Hz-lm-1.7B"
    logger.info(f"Initializing LM model ({lm_model})...")
    t0 = time.time()
    llm_handler = LLMHandler()
    lm_status, lm_success = llm_handler.initialize(
        checkpoint_dir=CHECKPOINT_DIR,
        lm_model_path=lm_model,
        backend=backend,
        device="auto",
        offload_to_cpu=auto_offload,
        dtype=None,
    )
    if not lm_success:
        logger.warning(f"LM initialization warning: {lm_status}. Proceeding with DiT direct generation.")
    else:
        logger.info(f"LM model ready in {time.time() - t0:.1f}s")

    # 4. Generation Loop with tqdm Progress Tracking
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting Generation: {TOTAL_ROUNDS} rounds x {BATCH_SIZE} tracks per round (Total: {TOTAL_ROUNDS * BATCH_SIZE} tracks)")
    logger.info(f"Prompt: {MUSIC_PROMPT}")
    logger.info(f"{'='*70}\n")

    progress_tracker = TqdmProgressTracker(total_rounds=TOTAL_ROUNDS, batch_size=BATCH_SIZE)
    all_generated_files = []
    overall_start_time = time.time()

    try:
        for round_idx in range(1, TOTAL_ROUNDS + 1):
            progress_tracker.start_round(round_idx)
            round_start = time.time()

            params = GenerationParams(
                task_type="text2music",
                caption=MUSIC_PROMPT,
                lyrics="[Instrumental]",
                instrumental=True,
                vocal_language="unknown",
                thinking=llm_handler.llm_initialized,
                inference_steps=8,
                guidance_scale=1.0,
                seed=-1,  # random seed each time
                enable_normalization=True,
                normalization_db=-1.0,
                duration=TARGET_DURATION,  # ~75s — fits in 8GB VRAM without timeout
            )

            config = GenerationConfig(
                batch_size=BATCH_SIZE,
                use_random_seed=True,
                audio_format="mp3",
                mp3_bitrate="320k",
            )

            try:
                result = generate_music(
                    dit_handler=dit_handler,
                    llm_handler=llm_handler,
                    params=params,
                    config=config,
                    save_dir=OUTPUT_DIR,
                    progress=progress_tracker.update_step_progress,
                )
            except TimeoutError as te:
                round_elapsed = time.time() - round_start
                progress_tracker.finish_round(0)
                tqdm.write(
                    f"⚠️  Round {round_idx}/{TOTAL_ROUNDS} timed out after {round_elapsed:.0f}s "
                    f"— skipping and continuing. ({te})"
                )
                continue
            except Exception as e:
                round_elapsed = time.time() - round_start
                progress_tracker.finish_round(0)
                tqdm.write(f"❌ Round {round_idx}/{TOTAL_ROUNDS} error after {round_elapsed:.0f}s: {e} — skipping.")
                continue

            round_elapsed = time.time() - round_start
            num_success = len(result.audios) if result.success else 0
            progress_tracker.finish_round(num_success)

            if result.success:
                tqdm.write(f"✅ Round {round_idx}/{TOTAL_ROUNDS} completed in {round_elapsed:.1f}s")
                for idx, audio in enumerate(result.audios, 1):
                    audio_path = audio.get("path", "(in-memory)")
                    seed = audio.get("params", {}).get("seed", "N/A")
                    tqdm.write(f"   ↳ Track {idx}/{BATCH_SIZE} [Seed: {seed}]: {audio_path}")
                    all_generated_files.append(audio_path)
            else:
                tqdm.write(f"❌ Round {round_idx}/{TOTAL_ROUNDS} failed: {result.status_message}")

    finally:
        progress_tracker.close()

    # Summary
    total_elapsed = time.time() - overall_start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"Generation Finished in {total_elapsed:.1f}s!")
    logger.info(f"Successfully generated {len(all_generated_files)} / {TOTAL_ROUNDS * BATCH_SIZE} tracks.")
    logger.info(f"Output folder: {OUTPUT_DIR}")
    for idx, path in enumerate(all_generated_files, 1):
        logger.info(f"  {idx:2d}. {path}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()
