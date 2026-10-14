"""
Ultra HDR gain-map reconstruction.

Implements the core merge step of the Ultra HDR / Adobe gain-map model:
combining an SDR base image with a gain map to recover an HDR image.

This module does NOT parse the Ultra HDR container format (the embedded
secondary JPEG + XMP metadata) -- it operates on already-decoded arrays.
Container parsing is a separate, larger piece of work; see the Ultra HDR
spec: https://developer.android.com/media/platform/hdr-image-format

Reference formula (simplified -- see module docstring note below):
    log2(HDR) = log2(SDR + offset_sdr) + gain_map * (max_log2 - min_log2) + min_log2
"""
import numpy as np
from rudin_osher_fatemi import denoise_tv_chambolle

from . import Image


def apply_gain_map(
    base,
    gain_map,
    headroom=3.0,
    min_headroom=0.0,
    offset_sdr=1 / 64,
    offset_hdr=1 / 64,
    denoise_weight=0.0,
):
    """
    Reconstruct an HDR image from an SDR base image and its gain map.

    :param base: The SDR base image, an :py:class:`~PIL.Image.Image` in
        mode "RGB".
    :param gain_map: The gain map, an :py:class:`~PIL.Image.Image` in
        mode "L". Will be resized to the base image's size if they differ,
        matching how renderers upsample the (usually lower-resolution)
        embedded gain map.
    :param headroom: ``max_log2`` -- the maximum extra stops of dynamic
        range the gain map can add, base-2 log scale. Encoders store this
        in XMP as ``hdrgm:HDRCapacityMax``.
    :param min_headroom: ``min_log2`` -- the minimum extra stops, usually
        0 (XMP ``hdrgm:HDRCapacityMin``).
    :param offset_sdr: Small offset added before taking log2 of the SDR
        channel, avoiding log(0). Adobe's gain map spec default is 1/64.
    :param offset_hdr: Matching offset on the HDR side.
    :param denoise_weight: If > 0, the gain map is denoised with ROF
        total-variation denoising (weight = this value) before being
        applied, reducing JPEG block artifacts in the reconstructed
        highlights. 0 disables denoising.
    :return: A new "RGB" :py:class:`~PIL.Image.Image` holding the HDR
        result, tone-mapped back into 8-bit range for display. (A future
        version could instead return a float array / mode "F" image for
        callers that want the untouched HDR values.)
    """
    if base.mode != "RGB":
        base = base.convert("RGB")
    if gain_map.mode != "L":
        gain_map = gain_map.convert("L")
    if gain_map.size != base.size:
        gain_map = gain_map.resize(base.size, Image.Resampling.BILINEAR)

    sdr = np.asarray(base, dtype=np.float64) / 255.0
    gmap = np.asarray(gain_map, dtype=np.float64) / 255.0

    if denoise_weight > 0:
        gmap = denoise_tv_chambolle(gmap, weight=denoise_weight)

    # Broadcast the single-channel gain map across R, G, B.
    gmap = gmap[:, :, np.newaxis]

    log2_hdr = (
        np.log2(sdr + offset_sdr)
        + gmap * (headroom - min_headroom)
        + min_headroom
    )
    hdr = np.exp2(log2_hdr) - offset_hdr
    hdr = np.clip(hdr, 0.0, None)

    # Tone-map back to [0, 1] for an 8-bit preview. Reinhard is a simple,
    # well-known operator; a real implementation would likely expose this
    # as a pluggable step rather than hardcoding it.
    tone_mapped = hdr / (1.0 + hdr)

    out = np.clip(tone_mapped * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")
