"""Behavioral simulation layer for browser automation.

Human-like mouse movement, scrolling, clicks, and timing to avoid
detection by anti-bot systems. All timing distributions are based on
empirical human behavior research.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random

from playwright.async_api import Page

logger = logging.getLogger(__name__)


def _gauss_clamp(mu: float, sigma: float, lo: float, hi: float) -> float:
    """Gaussian random with hard clamp."""
    return max(lo, min(hi, random.gauss(mu, sigma)))


def _lognormal(median: float, sigma: float = 0.5) -> float:
    """Log-normal variate with given median."""
    return random.lognormvariate(math.log(median), sigma)


async def human_move_to(page: Page, x: float, y: float) -> None:
    """Move mouse along a cubic Bezier curve to (x, y).

    Uses Fitts's Law for duration: T = 50 + 150 * log2(1 + D/W)
    with +/-20% human variance. Control points are offset perpendicular
    to the straight-line path with Gaussian noise.
    """
    current = await page.evaluate("() => ({x: window._mouseX || 0, y: window._mouseY || 0})")
    cx, cy = current.get("x", 0), current.get("y", 0)

    dx = x - cx
    dy = y - cy
    dist = math.hypot(dx, dy)

    if dist < 2:
        return

    # Fitts's Law timing (ms), assuming target width ~20px
    duration_ms = (50 + 150 * math.log2(1 + dist / 20)) * _gauss_clamp(1.0, 0.2, 0.7, 1.4)
    steps = max(8, int(duration_ms / 16))  # ~60fps

    # Bezier control points (perpendicular offset with noise)
    perp_x, perp_y = -dy / (dist + 1), dx / (dist + 1)
    offset1 = random.gauss(0, dist * 0.15)
    offset2 = random.gauss(0, dist * 0.10)

    cp1x = cx + dx * 0.3 + perp_x * offset1
    cp1y = cy + dy * 0.3 + perp_y * offset1
    cp2x = cx + dx * 0.7 + perp_x * offset2
    cp2y = cy + dy * 0.7 + perp_y * offset2

    for i in range(1, steps + 1):
        t = i / steps
        u = 1 - t
        # Cubic Bezier
        bx = u**3 * cx + 3 * u**2 * t * cp1x + 3 * u * t**2 * cp2x + t**3 * x
        by = u**3 * cy + 3 * u**2 * t * cp1y + 3 * u * t**2 * cp2y + t**3 * y

        await page.mouse.move(bx, by)
        await asyncio.sleep(random.uniform(0.012, 0.022))

    # Track position for next call
    await page.evaluate(f"() => {{ window._mouseX = {x}; window._mouseY = {y}; }}")


async def human_click(page: Page, selector: str) -> None:
    """Click an element with Gaussian coordinate distribution.

    Sigma = 1/6 of element dimension (99.7% within bounds).
    Clamped to 2px inset from element edges.
    """
    el = await page.wait_for_selector(selector, timeout=5000)
    if not el:
        raise ValueError(f"Element not found: {selector}")

    box = await el.bounding_box()
    if not box:
        raise ValueError(f"Element has no bounding box: {selector}")

    # Gaussian around center, clamped to 2px inset
    cx = box["x"] + box["width"] / 2
    cy = box["y"] + box["height"] / 2
    sigma_x = box["width"] / 6
    sigma_y = box["height"] / 6

    click_x = _gauss_clamp(cx, sigma_x, box["x"] + 2, box["x"] + box["width"] - 2)
    click_y = _gauss_clamp(cy, sigma_y, box["y"] + 2, box["y"] + box["height"] - 2)

    await human_move_to(page, click_x, click_y)
    await asyncio.sleep(random.uniform(0.05, 0.15))
    await page.mouse.click(click_x, click_y)


async def human_scroll(page: Page, direction: str = "down", amount: int = 0) -> None:
    """Scroll with inertial physics: initial burst then decaying tail.

    Parameters:
        direction: "down" or "up"
        amount: approximate total pixels (0 = random 300-800px)
    """
    if amount == 0:
        amount = random.randint(300, 800)

    sign = 1 if direction == "down" else -1

    # Initial burst: 3-6 large deltas
    burst_count = random.randint(3, 6)
    burst_delta = amount / (burst_count * 1.5)  # leave room for tail

    remaining = amount
    for _ in range(burst_count):
        delta = _gauss_clamp(burst_delta, burst_delta * 0.3, 40, 120)
        delta = min(delta, remaining)
        await page.mouse.wheel(0, sign * delta)
        remaining -= delta
        await asyncio.sleep(random.uniform(0.015, 0.080))

    # Decaying tail: 3-5 events, each ~half the previous
    tail_delta = max(10, remaining / 3)
    for _ in range(random.randint(3, 5)):
        if tail_delta < 5:
            break
        await page.mouse.wheel(0, sign * tail_delta)
        tail_delta *= random.uniform(0.4, 0.6)
        await asyncio.sleep(random.uniform(0.020, 0.100))

    # Occasional scroll-back (8% probability)
    if random.random() < 0.08:
        back_amount = random.randint(50, 150)
        await asyncio.sleep(random.uniform(0.3, 0.8))
        await page.mouse.wheel(0, -sign * back_amount)


async def dwell(min_s: float = 1.5, max_s: float = 20.0) -> None:
    """Log-normal dwell time, median ~4s."""
    t = _lognormal(4.0, 0.5)
    t = max(min_s, min(max_s, t))
    await asyncio.sleep(t)


async def pause(lo: float = 1.0, hi: float = 5.0) -> None:
    """Random inter-action pause."""
    await asyncio.sleep(random.uniform(lo, hi))


def session_duration_minutes() -> float:
    """Truncated Gaussian session duration: mean 25, range 15-35 min."""
    return _gauss_clamp(25.0, 5.0, 15.0, 35.0)
