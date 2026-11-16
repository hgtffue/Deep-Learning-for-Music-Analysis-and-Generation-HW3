"""Classifier-free guidance (CFG) for Stable Audio Open."""


def cfg(noise_pred_uncond, noise_pred_cond, guidance_scale):
    """Combine the unconditional and conditional model outputs of one denoising step.

    Slide formula:  D_cfg = D(c) + (w - 1) * (D(c) - D(phi))
    which is algebraically identical to the diffusers form
                    D_cfg = D(phi) + w * (D(c) - D(phi))
    We use the latter so the result is bit-identical to StableAudioPipeline.

    Args:
        noise_pred_uncond: tensor (1, C, L), model output with an empty (all-zero) condition.
        noise_pred_cond:   tensor (1, C, L), model output with the text prompt.
        guidance_scale:    float. 1.0 means no guidance, 0.0 means unconditional.
    Returns:
        tensor (1, C, L), the guided output passed to the scheduler.
    """
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
