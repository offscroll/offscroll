"""Behavioral simulation layer for browser-based crawlers.

Generates human-like mouse movements, scrolling, clicks, and timing
to avoid detection by anti-automation systems. All randomness uses
statistical distributions observed in real human interaction data.

Key design choices:
- Bézier curve mouse paths (not linear interpolation)
- Fitts's Law timing for mouse movements
- Inertial scroll physics with decaying deltas
- Gaussian click coordinate distribution (not dead center)
- Log-normal dwell times between actions
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass

from playwright.async_api import Page

logger = logging.getLogger(__name__)


@dataclass
class BehaviorConfig:
    """Tunable parameters for behavioral simulation."""

    # Mouse movement
    mouse_steps_per_move: int = 25
    fitts_a: float = 50.0  # Fitts's Law intercept (ms)
    fitts_b: float = 150.0  # Fitts's Law slope (ms)
    fitts_variance: float = 0.20  # ±20% human timing variance

    # Scrolling
    scroll_delta_min: int = 40
    scroll_delta_max: int = 120
    scroll_decay_steps: int = 4  # Inertial tail events
    scroll_decay_factor: float = 0.5
    scroll_inter_event_min_ms: int = 15
    scroll_inter_event_max_ms: int = 80
    scroll_back_probability: float = 0.08

    # Click coordinates
    click_sigma_fraction: float = 1 / 6  # Sigma = 1/6 of element dim
    click_edge_inset_px: int = 2

    # Timing
    dwell_median_s: float = 4.0  # Log-normal median
    dwell_sigma: float = 0.6  # Log-normal sigma (range ~1.5-20s)
    inter_action_min_s: float = 1.0
    inter_action_max_s: float = 5.0

    # Session
    session_mean_min: float = 25.0  # Truncated Gaussian mean (minutes)
    session_std_min: float = 5.0
    session_min_min: float = 15.0
    session_max_min: float = 35.0


def _cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Evaluate cubic Bézier curve at parameter t."""
    u = 1 - t
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    return (x, y)


def _generate_bezier_control_points(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Generate randomized Bézier control points for a natural mouse path.

    Control points are offset perpendicular to the straight-line path
    with Gaussian noise, producing curved trajectories.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy) or 1.0

    # Perpendicular direction
    perp_x = -dy / dist
    perp_y = dx / dist

    # Control point offsets: Gaussian, scaled to path length
    spread = dist * 0.3
    offset1 = random.gauss(0, spread)
    offset2 = random.gauss(0, spread)

    # Place control points at 1/3 and 2/3 along the path
    cp1 = (
        start[0] + dx / 3 + perp_x * offset1,
        start[1] + dy / 3 + perp_y * offset1,
    )
    cp2 = (
        start[0] + 2 * dx / 3 + perp_x * offset2,
        start[1] + 2 * dy / 3 + perp_y * offset2,
    )
    return cp1, cp2


def _fitts_time_ms(distance: float, width: float, cfg: BehaviorConfig) -> float:
    """Calculate movement time using Fitts's Law with human variance.

    T = a + b * log2(1 + D/W), with ±variance random jitter.
    """
    if width <= 0:
        width = 10.0
    base = cfg.fitts_a + cfg.fitts_b * math.log2(1 + distance / width)
    jitter = random.uniform(1 - cfg.fitts_variance, 1 + cfg.fitts_variance)
    return max(50.0, base * jitter)


async def move_mouse(
    page: Page,
    target_x: float,
    target_y: float,
    cfg: BehaviorConfig | None = None,
) -> None:
    """Move the mouse along a Bézier curve to (target_x, target_y).

    Uses Fitts's Law for total movement time, cubic Bézier for path,
    and per-step timing jitter for realistic cadence.
    """
    cfg = cfg or BehaviorConfig()

    # Get current mouse position (default to a random starting point)
    # Playwright doesn't expose current mouse pos, so we track it
    # via page._last_mouse_pos (set by us after each move).
    start = getattr(page, "_last_mouse_pos", (random.uniform(100, 500), random.uniform(100, 400)))
    end = (target_x, target_y)

    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    if distance < 2:
        # Already there
        return

    cp1, cp2 = _generate_bezier_control_points(start, end)
    total_time_ms = _fitts_time_ms(distance, 20.0, cfg)
    steps = cfg.mouse_steps_per_move
    step_time_ms = total_time_ms / steps

    for i in range(1, steps + 1):
        t = i / steps
        x, y = _cubic_bezier(start, cp1, cp2, end, t)
        await page.mouse.move(x, y)
        jitter = random.uniform(0.7, 1.3)
        await asyncio.sleep(step_time_ms * jitter / 1000)

    page._last_mouse_pos = end  # type: ignore[attr-defined]


async def human_click(
    page: Page,
    element,
    cfg: BehaviorConfig | None = None,
) -> None:
    """Click an element with Gaussian coordinate distribution.

    Moves the mouse naturally to the element, then clicks at a
    point drawn from a Gaussian centered on the element, not
    dead center.
    """
    cfg = cfg or BehaviorConfig()

    box = await element.bounding_box()
    if not box:
        # Element not visible; fall back to programmatic click
        await element.click()
        return

    # Gaussian click coordinates
    cx = box["x"] + box["width"] / 2
    cy = box["y"] + box["height"] / 2
    sigma_x = box["width"] * cfg.click_sigma_fraction
    sigma_y = box["height"] * cfg.click_sigma_fraction

    click_x = random.gauss(cx, sigma_x)
    click_y = random.gauss(cy, sigma_y)

    # Clamp to element bounds with inset
    inset = cfg.click_edge_inset_px
    click_x = max(box["x"] + inset, min(click_x, box["x"] + box["width"] - inset))
    click_y = max(box["y"] + inset, min(click_y, box["y"] + box["height"] - inset))

    await move_mouse(page, click_x, click_y, cfg)
    await page.mouse.click(click_x, click_y)


async def human_scroll(
    page: Page,
    direction: str = "down",
    cfg: BehaviorConfig | None = None,
) -> None:
    """Scroll with inertial physics: initial burst + decaying tail.

    Occasionally scrolls back (8% default) to simulate re-reading.
    """
    cfg = cfg or BehaviorConfig()

    # Occasional scroll-back
    if direction == "down" and random.random() < cfg.scroll_back_probability:
        logger.debug("Scroll-back triggered")
        await human_scroll(page, "up", cfg)
        await asyncio.sleep(random.uniform(0.5, 2.0))

    sign = -1 if direction == "down" else 1  # wheel delta: negative = scroll down
    initial_delta = random.randint(cfg.scroll_delta_min, cfg.scroll_delta_max)

    # Initial burst
    await page.mouse.wheel(0, sign * initial_delta)
    await asyncio.sleep(
        random.randint(cfg.scroll_inter_event_min_ms, cfg.scroll_inter_event_max_ms) / 1000
    )

    # Decaying tail events
    delta = initial_delta
    for _ in range(random.randint(2, cfg.scroll_decay_steps + 1)):
        delta = int(delta * cfg.scroll_decay_factor)
        if delta < 5:
            break
        await page.mouse.wheel(0, sign * delta)
        await asyncio.sleep(
            random.randint(cfg.scroll_inter_event_min_ms, cfg.scroll_inter_event_max_ms) / 1000
        )


async def dwell(cfg: BehaviorConfig | None = None) -> None:
    """Wait a log-normal dwell time, simulating reading/scanning."""
    cfg = cfg or BehaviorConfig()
    mu = math.log(cfg.dwell_median_s)
    t = random.lognormvariate(mu, cfg.dwell_sigma)
    t = max(1.0, min(t, 30.0))  # Clamp to reasonable range
    logger.debug("Dwelling for %.1fs", t)
    await asyncio.sleep(t)


async def inter_action_pause(cfg: BehaviorConfig | None = None) -> None:
    """Short pause between actions (uniform distribution)."""
    cfg = cfg or BehaviorConfig()
    t = random.uniform(cfg.inter_action_min_s, cfg.inter_action_max_s)
    await asyncio.sleep(t)


def session_duration_minutes(cfg: BehaviorConfig | None = None) -> float:
    """Generate a randomized session duration from truncated Gaussian."""
    cfg = cfg or BehaviorConfig()
    while True:
        d = random.gauss(cfg.session_mean_min, cfg.session_std_min)
        if cfg.session_min_min <= d <= cfg.session_max_min:
            return d
