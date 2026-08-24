"""
ACE-Step V1.5 — Batch Music Generation Script
Generates 5 rounds of 2-batch music tracks (10 tracks total) for a given description.
"""

import os
import sys
import time
from pathlib import Path
from loguru import logger

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

TOTAL_ROUNDS = 5
BATCH_SIZE = 2


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

    # 4. Generation Loop: 5 rounds x 2 batches = 10 tracks
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting Generation: {TOTAL_ROUNDS} rounds x {BATCH_SIZE} tracks per round (Total: {TOTAL_ROUNDS * BATCH_SIZE} tracks)")
    logger.info(f"Prompt: {MUSIC_PROMPT}")
    logger.info(f"{'='*70}\n")

    all_generated_files = []
    overall_start_time = time.time()

    for round_idx in range(1, TOTAL_ROUNDS + 1):
        logger.info(f"--- Round {round_idx}/{TOTAL_ROUNDS} (Generating {BATCH_SIZE} tracks) ---")
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
        )

        config = GenerationConfig(
            batch_size=BATCH_SIZE,
            use_random_seed=True,
            audio_format="mp3",
            mp3_bitrate="320k",
        )

        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=params,
            config=config,
            save_dir=OUTPUT_DIR,
        )

        round_elapsed = time.time() - round_start

        if result.success:
            logger.info(f"Round {round_idx}/{TOTAL_ROUNDS} completed in {round_elapsed:.1f}s")
            for idx, audio in enumerate(result.audios, 1):
                audio_path = audio.get("path", "(in-memory)")
                seed = audio.get("params", {}).get("seed", "N/A")
                logger.info(f"  Track {idx}/{BATCH_SIZE} [Seed: {seed}]: {audio_path}")
                all_generated_files.append(audio_path)
        else:
            logger.error(f"Round {round_idx}/{TOTAL_ROUNDS} failed: {result.status_message}")

        logger.info("")

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
