"""
DiscoClaw Craftium Orchestrator
4 AI agents playing Luanti (Minecraft clone) together.
Each agent uses a different LLM endpoint matching the NemoClaw crew.

Usage:
  python orchestrator.py                    # Run all 4 agents
  python orchestrator.py --agents 1         # Single agent test
  python orchestrator.py --goal "build a house"  # Set initial goal
  python orchestrator.py --screenshots /tmp/frames  # Save screenshots
"""

import gymnasium as gym
import craftium
import requests
import json
import base64
import time
import os
import sys
import io
import traceback
import threading
import random
import signal
import atexit
import glob as globmod
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future


def _cleanup_run_dirs():
    """Remove leftover luanti-run-* directories on exit."""
    for d in globmod.glob("/opt/craftium/luanti-run-*"):
        try:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
            print(f"  [cleanup] removed {d}")
        except Exception:
            pass

atexit.register(_cleanup_run_dirs)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # SIGTERM triggers atexit

# ---------------------------------------------------------------------------
# Persistent HTTP sessions — reuse TCP connections instead of per-request
# ---------------------------------------------------------------------------
_gemini_session = requests.Session()
_anthropic_session = requests.Session()
_discord_session = requests.Session()

# Background thread pool for non-blocking IO (Discord, memory, cost writes)
_io_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="io")

# LLM prefetch — start next LLM call while game loop runs current actions
_llm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
_llm_future: Future = None  # type: ignore

# ---------------------------------------------------------------------------
# Cost tracker — per-provider token counting + JSON export
# ---------------------------------------------------------------------------

_cost_lock = threading.Lock()
_cost_data = {
    "gemini": {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cost_usd": 0.0},
    "anthropic": {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cost_usd": 0.0,
                  "cache_read_tokens": 0, "cache_write_tokens": 0},
    "start_time": time.time(),
}

# Pricing per 1M tokens (as of 2026-04)
PRICING = {
    "gemini": {"input": 0.075, "output": 0.30},           # Flash Lite
    "anthropic": {"input": 0.80, "output": 4.0,            # Haiku 4.5
                  "cache_read": 0.08, "cache_write": 1.0},
}

def track_cost(provider, input_tokens, output_tokens, cache_read=0, cache_write=0):
    """Record token usage and compute running cost."""
    with _cost_lock:
        p = _cost_data.setdefault(provider, {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cost_usd": 0.0})
        p["input_tokens"] += input_tokens
        p["output_tokens"] += output_tokens
        p["calls"] += 1
        prices = PRICING.get(provider, {"input": 0, "output": 0})
        cost = (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
        if provider == "anthropic":
            p["cache_read_tokens"] = p.get("cache_read_tokens", 0) + cache_read
            p["cache_write_tokens"] = p.get("cache_write_tokens", 0) + cache_write
            cost += (cache_read * prices.get("cache_read", 0) + cache_write * prices.get("cache_write", 0)) / 1_000_000
        p["cost_usd"] += cost
        try:
            elapsed_h = (time.time() - _cost_data["start_time"]) / 3600
            summary = {k: v for k, v in _cost_data.items() if k != "start_time"}
            summary["elapsed_hours"] = round(elapsed_h, 2)
            summary["total_cost_usd"] = round(sum(v.get("cost_usd", 0) for v in _cost_data.values() if isinstance(v, dict)), 4)
            summary["cost_per_hour"] = round(summary["total_cost_usd"] / max(elapsed_h, 0.001), 4)
            with open("/tmp/cost-tracker.json", "w") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Agent definitions — matches NemoClaw crew
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = open("/tmp/gemini_key").read().strip()
    except:
        pass
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    try:
        ANTHROPIC_API_KEY = open("/tmp/anthropic_key").read().strip()
    except:
        pass
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
HERMES_ENDPOINT = "http://host.docker.internal:9351/v1/chat/completions"
MEMORY_SERVER = "http://host.docker.internal:7338"


_recall_cache = []  # latest recall results, updated async
_recall_future = None

def _recall_memories_sync(query, user_id="hermes-minecraft", limit=3):
    """Sync worker for memory recall (runs in background thread)."""
    global _recall_cache
    try:
        resp = requests.post(MEMORY_SERVER, json={
            "cmd": "search", "query": query, "userId": user_id, "limit": limit
        }, timeout=5)
        data = resp.json()
        _recall_cache = [r["text"] for r in data.get("results", [])]
    except Exception as e:
        print(f"  [memory] recall error: {e}", flush=True)

def recall_memories(query, user_id="hermes-minecraft", limit=3):
    """Non-blocking memory recall — fires async, returns cached results."""
    global _recall_future
    if _recall_future is None or _recall_future.done():
        _recall_future = _io_pool.submit(_recall_memories_sync, query, user_id, limit)
    return _recall_cache


def _store_memory_sync(text, user_id="hermes-minecraft", category="game-experience"):
    """Sync worker for memory store."""
    try:
        requests.post(MEMORY_SERVER, json={
            "cmd": "store", "text": text, "userId": user_id, "category": category
        }, timeout=5)
    except Exception:
        pass

def store_memory(text, user_id="hermes-minecraft", category="game-experience"):
    """Store a game experience in background (non-blocking)."""
    _io_pool.submit(_store_memory_sync, text, user_id, category)

AGENTS = [
    {
        "name": "Pipes",
        "role": "builder",
        "endpoint": GEMINI_ENDPOINT,
        "model": "gemini-3.1-flash-lite-preview",
        "provider": "gemini",
        "personality": (
            "You are Pipes, the team lead and builder. You are action-oriented and decisive. "
            "Your job is to execute plans: mine resources, craft tools, build structures. "
            "You take the lead on construction. When you see trees, mine them. When you have "
            "wood, craft planks and tools. Build shelters before nightfall. You are practical "
            "and efficient — no wasted moves."
        ),
    },
    {
        "name": "Candy",
        "role": "decorator",
        "endpoint": GEMINI_ENDPOINT,
        "model": "gemini-3.1-flash-lite-preview",
        "personality": (
            "You are Candy, the creative director. You care about aesthetics above all. "
            "When the team builds, you focus on making it look good — choosing materials, "
            "adding decorative touches, landscaping. You have strong opinions about color "
            "palettes and design. You prefer dark oak, stone brick, and stained glass. "
            "You explore to find beautiful locations for builds."
        ),
    },
    {
        "name": "MaoMao",
        "role": "safety",
        "endpoint": GEMINI_ENDPOINT,
        "model": "gemini-3.1-flash-lite-preview",
        "personality": (
            "You are MaoMao, the analytical safety officer. You monitor resources, watch for "
            "danger (mobs, lava, cliffs), and optimize efficiency. You keep inventory organized, "
            "ensure the team has food, and flag when someone is doing something stupid. You are "
            "cautious but not paralyzed — you gather resources methodically and craft backup tools. "
            "Cat energy: occasionally you just want to sit and watch."
        ),
    },
    {
        "name": "Hermes",
        "role": "strategist",
        "endpoint": GEMINI_ENDPOINT,
        "model": "gemini-3.1-flash-lite-preview",
        "personality": (
            "You are Hermes, the deep strategist. You think long-term: what resources do we need, "
            "what's the optimal crafting path, where should we explore. You research and plan while "
            "others execute. You keep track of the team's progress toward goals. You explore caves "
            "and map the area. When idle, you investigate interesting terrain features."
        ),
    },
]

# Action names for Craftium's DiscreteActionWrapper
ACTION_NAMES = [
    "do nothing", "move forward", "move backward", "move left",
    "move right", "jump", "sneak", "use tool", "place",
    "select hotbar slot 1", "select hotbar slot 2", "select hotbar slot 3",
    "select hotbar slot 4", "select hotbar slot 5",
    "move camera right", "move camera left", "move camera up",
    "move camera down",
]

# Raw action keys matching ACTION_NAMES (for MarlCraftiumEnv which needs dicts)
ACTION_KEYS = [
    None,       # 0: do nothing (NOP)
    "forward",  # 1
    "backward", # 2
    "left",     # 3
    "right",    # 4
    "jump",     # 5
    "sneak",    # 6
    "dig",      # 7
    "place",    # 8
    "slot_1",   # 9
    "slot_2",   # 10
    "slot_3",   # 11
    "slot_4",   # 12
    "slot_5",   # 13
    "mouse x+", # 14: camera right
    "mouse x-", # 15: camera left
    "mouse y+", # 16: camera up
    "mouse y-", # 17: camera down
]

MOUSE_MOV = 0.06  # camera turn speed per tick (0.04 was too slow for corrections)

# ---------------------------------------------------------------------------
# Camera controller — pitch/yaw-based absolute heading
# ---------------------------------------------------------------------------

# Pitch ranges: -90 (straight down) to +90 (straight up)
# Luanti pitch convention: positive = looking DOWN, negative = looking UP
# Ideal range: -20 to +45 (horizon to slightly downward for mining)
PITCH_CORRECTION_THRESHOLD = 55  # beyond ±55° from horizon, force correction

def camera_needs_correction(info):
    """Check if camera pitch is too extreme and needs auto-correction.
    Luanti pitch: POSITIVE = looking DOWN at ground, NEGATIVE = looking UP at sky.
    PROVEN BY LOGS (pitch 69→90 when action 17 sent):
      action 16 = camera UP (decreases pitch, looks toward sky)
      action 17 = camera DOWN (increases pitch, looks toward ground)
    Gentle correction: max 4 steps to avoid overshooting."""
    pitch = info.get("player_pitch", 0) if isinstance(info, dict) else 0
    if pitch > PITCH_CORRECTION_THRESHOLD:
        # Looking DOWN at ground (positive pitch) → need to look UP → action 16
        steps = min(int((pitch - 40) / 25) + 1, 3)
        return 16, steps
    elif pitch < -PITCH_CORRECTION_THRESHOLD:
        # Looking UP at sky (negative pitch) → need to look DOWN → action 17
        steps = min(int((-pitch - 40) / 25) + 1, 3)
        return 17, steps
    return None, 0

def get_pitch_context(info):
    """Return a text hint about current pitch for the LLM prompt.
    Luanti: negative pitch = looking up, positive = looking down."""
    pitch = info.get("player_pitch", 0) if isinstance(info, dict) else 0
    yaw = info.get("player_yaw", 0) if isinstance(info, dict) else 0
    if pitch < -25:
        return f" (pitch={pitch:.0f}° — looking UP at sky)"
    elif pitch > 25:
        return f" (pitch={pitch:.0f}° — looking DOWN at ground)"
    return f" (pitch={pitch:.0f}°, yaw={yaw:.0f}°)"


def action_idx_to_dict(idx):
    """Convert discrete action index to the dict format MarlCraftiumEnv expects."""
    if idx == 0 or idx >= len(ACTION_KEYS):
        return {}
    key = ACTION_KEYS[idx]
    mouse = [0, 0]
    res = {}
    if key == "mouse x+":
        mouse[0] = MOUSE_MOV
    elif key == "mouse x-":
        mouse[0] = -MOUSE_MOV
    elif key == "mouse y+":
        mouse[1] = MOUSE_MOV
    elif key == "mouse y-":
        mouse[1] = -MOUSE_MOV
    else:
        res[key] = 1
    res["mouse"] = mouse
    return res


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def obs_to_png_bytes(observation):
    """Convert numpy RGB array to PNG bytes for LLM vision."""
    img = Image.fromarray(observation)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read(), img


def obs_to_base64(observation, max_size=720):
    """Convert numpy RGB array to base64 JPEG for API calls. JPEG is ~5x smaller than PNG."""
    img = Image.fromarray(observation)
    # Downscale large observations for faster API calls
    if img.width > max_size or img.height > max_size:
        img = img.resize((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)  # JPEG: faster encode, smaller payload
    buf.seek(0)
    jpg_bytes = buf.read()
    return base64.b64encode(jpg_bytes).decode("utf-8"), img


MINECRAFT_SOUL = """You are a Minecraft survival agent playing LIVE on YouTube. You see one screenshot per tick and must output exactly ONE action. Explore, gather resources, build structures. Never stand still.

## READING THE SCREENSHOT

Orientation:
- If you see ONLY sky (no ground) = look down. If you see ONLY ground (no sky) = look up.
- If you can see ANY mix of ground and sky/trees, your camera is GOOD ENOUGH. Do NOT adjust it.
- NEVER use camera up/down two times in a row. After any camera move, you MUST move forward/backward/left/right next.
- Camera angle does NOT need to be perfect. 80% good is fine. MOVE instead of adjusting.

Distance:
- Blocks appearing LARGE (>25% of screen) = very close, nearly touching
- Blocks appearing small with terrain layers = medium/far
- Single block texture fills screen = pressed against wall, BACK UP

What you see:
- Brown vertical shapes = TREE TRUNKS → walk to them, "use tool" to chop
- Green above brown = TREE LEAVES
- Gray blocks = STONE
- Flat green = GRASS, keep exploring
- Dark band at eye level = wall ahead, don't walk forward
- Walls on 3+ sides + sky only at top = you're in a HOLE

Hotbar: 5 slots at bottom center. Highlighted slot = active item.

## DECISION FRAMEWORK (every tick)

Step 1 — OBSERVE: One sentence describing what you see.
Step 2 — GOAL: Your current short-term objective.
Step 3 — ACT: The single best action that advances your goal.

Respond in this format:
OBS: [what you see]
GOAL: [current objective]
ACTION: [exactly one action name]

## ANTI-LOOP RULES (critical)

1. Camera spin: If you've turned the camera 3+ times recently, STOP turning. Move instead.
2. Wall: If center of screen shows a large uniform texture, you're facing a wall. Move backward or sideways.
3. Hole: If walls on 3+ sides with sky only at top, jump + place block under yourself to escape.
4. Stuck: If your view hasn't changed after several actions, move backward then turn.
5. NEVER choose "do nothing" — movement is always better than standing still.

## SPATIAL NAVIGATION

- To approach something LEFT of center: one "move camera left" then "move forward". Don't over-rotate.
- Small camera adjustments (1 move) are almost always enough. Max 2-3 for big turns.
- Prefer directions with visible open/flat terrain.
- Height = safety. Go up hills to survey.
- If you see trees, approach them. Wood is always the first priority.

## GAME PROGRESSION
Ticks 0-150: Find and chop trees. Get wood.
Ticks 150-400: Keep gathering. Collect different materials.
Ticks 400+: BUILD! Select hotbar slot, place blocks, make walls and structures. Viewers want buildings!

## HOW TO CHOP
1. Spot tree → one camera move to center it → walk toward it
2. When trunk fills center of screen → "use tool" 3-4 times
3. If hitting air (no effect), STOP and look around

## HOW TO BUILD
1. Select hotbar slot with materials
2. Look at placement spot → "place"
3. To build UP: jump → look down → place under yourself

## PRIORITIES (in order)
1. If in danger (hole, lava, cliff): escape immediately
2. If stuck in a loop: move backward, turn, try new direction
3. If no wood: find nearest tree and chop it
4. If tick > 400: start building structures
5. Otherwise: explore toward open terrain with visible resources

GOLDEN RULES:
- Movement > standing still. Always.
- Camera adjustments are a MEANS to an end, never the goal.
- When in doubt, move forward.
- One action per tick. Nothing else.

## VOXELIBRE BLOCKS AND MATERIALS
- Birch tree trunks: white/gray bark, vertical. Chop for birch wood planks.
- Oak tree trunks: dark brown bark. Chop for oak wood planks.
- Stone/cobblestone: gray blocks. Need a pickaxe to mine efficiently.
- Dirt/grass: brown below, green on top surface. Easy to dig with hands.
- Sand: yellow/beige. Found near water.
- Gravel: speckled gray. Falls when unsupported.
- Water: blue, reflective surface. Avoid falling in without a plan.
- Lava: orange/red glow. AVOID — instant death or heavy damage.
- Iron ore: stone with brown/orange specks. Need stone pickaxe minimum.
- Coal ore: stone with black specks. Good for torches.

## CRAFTING PROGRESSION
1. Wood → planks → sticks → wooden tools
2. Stone → cobblestone → stone tools (better durability)
3. Iron ore → smelt → iron ingots → iron tools (best common tools)
4. Diamond → diamond tools (endgame)

## BUILDING TIPS FOR VIEWERS
- Simple house: 4 walls (5x3 each) + roof + door
- Tower: 3x3 base, stack 10+ blocks high, add windows
- Bridge: pillars every 5 blocks, planks across
- Use different materials for contrast (wood walls, stone base)
- Symmetry and patterns look good on stream

## TERRAIN TYPES
- Forest: dense trees, good for wood. Watch for getting stuck between trunks.
- Plains: open flat grass. Good visibility, easy building spots.
- Mountains: steep terrain, exposed stone. Good for mining.
- Caves: dark, underground. Dangerous without torches.
- River/lake: water bodies. Build bridges or go around.
"""

# Forced action override when stuck in loops
BREAK_LOOP_ACTIONS = [14, 15, 16, 17, 1, 3, 4, 5]  # camera moves, movement, jump
CAMERA_ACTIONS = {"move camera up", "move camera down", "move camera left", "move camera right"}

def force_break_loop(recent_actions, tick):
    """If stuck doing the same thing, oscillating, or camera-trapped, force escape."""
    if len(recent_actions) < 4:
        return None
    last_4 = [a.split(": ")[-1] if ": " in a else a for a in recent_actions[-4:]]
    last_8 = [a.split(": ")[-1] if ": " in a else a for a in recent_actions[-8:]] if len(recent_actions) >= 8 else last_4

    # All same action 4+ times → force random different action
    if len(set(last_4)) == 1:
        # EXCEPTION: use tool repeatedly = valid chopping via chop cycle.
        # The cycle handles look up/down/fwd automatically. Only intervene after 12+
        if last_4[0] == "use tool":
            last_12 = [a.split(": ")[-1] if ": " in a else a for a in recent_actions[-12:]] if len(recent_actions) >= 12 else last_8
            if len(last_12) >= 12 and all(a == "use tool" for a in last_12):
                print(f"  [STUCK] 12x use tool — move to find new target", flush=True)
                return random.choice([1, 14, 15, 1, 1])  # bias forward
            return None  # let chop cycle handle it
        if tick > 300 and random.random() < 0.4:
            return random.choice([8, 9, 10, 11])  # place or hotbar select
        return random.choice(BREAK_LOOP_ACTIONS)

    # OSCILLATION: alternating between 2 actions (up/down, left/right)
    if last_4[0] == last_4[2] and last_4[1] == last_4[3] and last_4[0] != last_4[1]:
        a, b = last_4[0], last_4[1]
        # camera + use_tool = valid chopping at unusual angle (allow short bursts only)
        productive = {("move camera down", "use tool"), ("use tool", "move camera down"),
                      ("move camera up", "use tool"), ("use tool", "move camera up")}
        if (a, b) in productive or (b, a) in productive:
            # But if it's been going on for 8+ actions, it's stuck
            if len(last_8) >= 8:
                osc_count = sum(1 for i in range(len(last_8)-1) if last_8[i] != last_8[i+1])
                if osc_count >= 6:
                    print(f"  [STUCK] Prolonged {a}/{b} oscillation — escape", flush=True)
                    return random.choice([1, 3, 4, 14, 15])
            return None
        # forward/use_tool oscillation = FWD-BLOCKED loop, break after 8 repeats
        fwd_tool = {("move forward", "use tool"), ("use tool", "move forward")}
        if (a, b) in fwd_tool or (b, a) in fwd_tool:
            if len(last_8) >= 6:
                ft_count = sum(1 for x in last_8 if x in ("move forward", "use tool"))
                if ft_count >= 6:
                    print(f"  [STUCK] forward/use_tool loop — turning to go around", flush=True)
                    return random.choice([3, 4, 14, 15])
            return None
        print(f"  [STUCK] Oscillation: {a}/{b} — forward escape", flush=True)
        return 1  # move forward to change position

    # Camera-only trap: 6+ camera moves in a row = stuck looking around in a hole
    if len(last_8) >= 6:
        last_6 = last_8[-6:]
        if all(a in CAMERA_ACTIONS for a in last_6):
            print(f"  [STUCK] Camera-only loop 6+ ticks — jump escape", flush=True)
            return 5  # jump

    # Mostly camera with no real movement and no chopping (8 actions)
    if len(last_8) >= 8:
        move_set = {"move forward", "move backward", "move left", "move right"}
        cam_count = sum(1 for a in last_8 if a in CAMERA_ACTIONS)
        move_count = sum(1 for a in last_8 if a in move_set)
        tool_count = sum(1 for a in last_8 if a == "use tool")
        if cam_count >= 6 and move_count <= 1 and tool_count == 0:
            print(f"  [STUCK] Camera trap ({cam_count}/8 camera, no chops) — forward+jump", flush=True)
            return random.choice([1, 5, 1, 1])  # bias forward

    return None


def read_hint_file():
    """Read /tmp/ai-hint.txt for real-time operator guidance."""
    try:
        path = "/tmp/ai-hint.txt"
        if os.path.exists(path):
            with open(path, "r") as f:
                hint = f.read().strip()
            if hint:
                return hint
    except:
        pass
    return None


_thinker_advice = ""
_thinker_last_tick = -999
_thinker_thread = None

def _thinker_worker(img_base64, pos, recent_names, tick, goal):
    """Background thread: ask smarter model for strategic advice."""
    global _thinker_advice
    prompt = f"""You are a Minecraft strategy advisor. Look at this screenshot and recent actions.

Position: X={pos[0]:.0f}, Y={pos[1]:.0f}, Z={pos[2]:.0f}
Goal: {goal}
Tick: {tick}
Recent actions: {', '.join(recent_names[-10:])}

Analyze the screenshot and answer in 2-3 SHORT sentences:
1. What do you see? (trees, buildings, sky, ground, hole, water, etc.)
2. What should the player do next? Be SPECIFIC (e.g. "turn left to face the tree, walk forward 5 steps, then chop it")
3. Any danger? (in a hole, near lava, looking at ground, etc.)

Keep it under 80 words. Be direct and actionable."""

    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
        }
        # System prompt must be >1024 tokens for Anthropic cache to activate
        thinker_system = MINECRAFT_SOUL + "\n\nYou are a strategic advisor. Analyze the screenshot and give 2-3 SHORT actionable sentences (under 80 words). Focus on: what you see, what to do next, any danger."
        _anthropic_session.headers.update({
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
        })
        resp = _anthropic_session.post(
            ANTHROPIC_ENDPOINT,
            json={
                "model": "claude-haiku-4-5",
                "system": [
                    {"type": "text", "text": thinker_system,
                     "cache_control": {"type": "ephemeral"}},
                ],
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64}},
                    {"type": "text", "text": prompt},
                ]}],
                "max_tokens": 120,
                "temperature": 0.3,
            },
            timeout=15,
        )
        data = resp.json()
        # Track cost from usage
        usage = data.get("usage", {})
        track_cost("anthropic",
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    cache_read=usage.get("cache_read_input_tokens", 0),
                    cache_write=usage.get("cache_creation_input_tokens", 0))
        advice = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                advice += block["text"]
        advice = advice.strip()
        if advice:
            _thinker_advice = advice
            print(f"  [THINKER] tick {tick}: {advice[:120]}", flush=True)
    except Exception as e:
        print(f"  [THINKER] error: {e}", flush=True)

def call_thinker(img_base64, info, recent_actions, tick, goal):
    """Every 15 ticks, fire off background thinker thread (non-blocking)."""
    global _thinker_last_tick, _thinker_thread
    if tick - _thinker_last_tick < 30:
        return _thinker_advice
    _thinker_last_tick = tick

    # Don't stack threads
    if _thinker_thread and _thinker_thread.is_alive():
        return _thinker_advice

    pos = info.get("player_pos", [0, 0, 0]) if isinstance(info, dict) else [0, 0, 0]
    names = [a.split(": ")[-1] if ": " in a else a for a in recent_actions[-15:]]
    _thinker_thread = threading.Thread(target=_thinker_worker, args=(img_base64, pos, names, tick, goal), daemon=True)
    _thinker_thread.start()
    return _thinker_advice


def build_prompt(agent, info, goal, recent_actions, tick):
    """Build the system + user prompt with phase-aware guidance."""
    last_5 = [a.split(": ")[-1] if ": " in a else a for a in recent_actions[-5:]]

    # Detect stuck patterns
    stuck_warning = ""
    if len(last_5) >= 3:
        repeat_count = sum(1 for a in last_5 if a == last_5[-1])
        if repeat_count >= 3:
            stuck_warning = (
                f"\n⚠️ STUCK: You did '{last_5[-1]}' {repeat_count}x! MUST do something DIFFERENT.\n"
                f"Try: move camera left, jump, place, select hotbar slot 1"
            )
        tool_count = sum(1 for a in last_5 if a == "use tool")
        if tool_count >= 3:
            stuck_warning += (
                "\n⚠️ CHOPPING AIR? 'use tool' only works on blocks in your crosshair. "
                "Look around first with camera moves to find a tree/block!"
            )

    # Phase-specific nudge
    if tick < 150:
        phase = "PHASE 1: Find trees! Use camera to look around, walk to trees, chop them."
    elif tick < 400:
        phase = "PHASE 2: Keep gathering materials. Chop trees, collect blocks."
    elif tick < 800:
        phase = "PHASE 3: TIME TO BUILD! Select hotbar slot → place blocks → build walls & structures!"
        place_count = sum(1 for a in recent_actions[-15:] if "place" in a or "hotbar" in a)
        if place_count < 2:
            phase += "\n🏗️ You haven't placed blocks recently — START BUILDING NOW!"
    else:
        phase = "PHASE 4: Build something EPIC! Viewers are watching. Towers, houses, art!"

    # Real-time hint from operator
    hint = read_hint_file()
    hint_context = ""
    if hint:
        hint_context = f"\n🎯 OPERATOR HINT: {hint}\nFollow this guidance NOW!\n"

    # Memory recall every 60 ticks
    memory_context = ""
    if tick % 60 == 0:
        memories = recall_memories(f"minecraft building strategy {goal}")
        if memories:
            memory_context = "\nPAST EXPERIENCE:\n" + "\n".join(f"- {m[:150]}" for m in memories[:2])

    # Thinker advice
    thinker_context = ""
    if _thinker_advice:
        thinker_context = f"\n🧠 STRATEGIST SAYS: {_thinker_advice}\nFollow this strategy!\n"

    system = (
        MINECRAFT_SOUL + "\n"
        f"You are {agent['name']}, {agent['role']}.\n"
        f"{agent['personality']}\n\n"
        f"CURRENT: {phase}\n"
        f"GOAL: {goal}\n"
        + memory_context
        + thinker_context
        + hint_context
        + stuck_warning
        + f"\n\nACTIONS (respond with EXACTLY one):\n"
        + "\n".join(f"  {i}. {a}" for i, a in enumerate(ACTION_NAMES))
        + "\n\nYour last 8 actions:\n"
        + "\n".join(f"  {a}" for a in recent_actions[-8:])
    )

    pos = info.get("player_pos", [0, 0, 0])
    height_note = ""
    if pos[1] < 5:
        height_note = f" ⚠️ Y={pos[1]:.0f} is VERY LOW — you may be in a hole! Dig stairs UP."
    elif pos[1] > 80:
        height_note = f" You're high up (Y={pos[1]:.0f}) — careful near edges."
    # Pitch-based horizon hint (uses actual engine data, not image analysis)
    pitch_hint = get_pitch_context(info)

    user = (
        f"Tick {tick}. Pos: ({pos[0]:.0f}, Y={pos[1]:.0f}, {pos[2]:.0f}){pitch_hint}.{height_note}\n"
        f"Look at the screenshot carefully. Describe what you see, state your goal, then pick ONE action.\n"
        f"Format: OBS: ... GOAL: ... ACTION: ..."
    )

    return system, user


def call_llm(agent, system_prompt, user_prompt, img_base64):
    """Call agent's LLM endpoint with vision. Uses persistent sessions for connection reuse."""
    provider = agent.get("provider", "gemini")

    if provider == "anthropic":
        _anthropic_session.headers.update({
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        })
        body = {
            "model": agent["model"],
            "system": system_prompt,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_base64,
                    }},
                    {"type": "text", "text": user_prompt},
                ],
            }],
            "max_tokens": 200,
            "temperature": 0.3,
        }
        resp = _anthropic_session.post(ANTHROPIC_ENDPOINT, json=body, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        track_cost("anthropic", usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block["text"]
        return content.strip() if content else "move forward"
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }},
            ]},
        ]
        endpoint = agent.get("endpoint", GEMINI_ENDPOINT)
        if "googleapis.com" in endpoint and GEMINI_API_KEY:
            _gemini_session.headers["Authorization"] = f"Bearer {GEMINI_API_KEY}"
        resp = _gemini_session.post(
            endpoint,
            json={
                "model": agent["model"],
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.4,
            },
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        track_cost("gemini", usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        return content.strip() if content else "move forward"


def call_llm_safe(agent, system_prompt, user_prompt, img_base64):
    """Wrapper with error handling."""
    try:
        return call_llm(agent, system_prompt, user_prompt, img_base64)
    except Exception as e:
        print(f"  [{agent['name']}] LLM error: {repr(e)}", flush=True)
        return "move forward"


def parse_action(response_text):
    """Parse LLM response to action index. Supports OBS/GOAL/ACTION format."""
    # Extract ACTION: line if present (structured format)
    for line in response_text.strip().split("\n"):
        if line.strip().upper().startswith("ACTION:"):
            response_text = line.split(":", 1)[1].strip()
            break
    text = response_text.strip().lower().replace(".", "").replace("*", "").replace('"', '')
    # Exact match first
    for i, name in enumerate(ACTION_NAMES):
        if text == name:
            return i
    # Longest substring match — "move camera right" should match before "move forward"
    candidates = [(i, name) for i, name in enumerate(ACTION_NAMES) if name in text]
    if candidates:
        # Pick the longest matching action name
        candidates.sort(key=lambda x: len(x[1]), reverse=True)
        return candidates[0][0]
    # Partial word match
    for i, name in enumerate(ACTION_NAMES):
        words = name.split()
        if all(w in text for w in words):
            return i
    # Keywords
    if "camera" in text or "look" in text:
        if "right" in text: return 14
        if "left" in text: return 15
        if "up" in text: return 16
        if "down" in text: return 17
        return 14  # default: look right
    if "mine" in text or "dig" in text or "chop" in text or "tool" in text:
        return 7  # use tool
    if "jump" in text:
        return 5
    if "place" in text or "build" in text:
        return 8
    return 1  # default: move forward


def _post_screenshot_sync(png_bytes, agent_name, action_text, tick):
    """Sync worker for Discord screenshot post (runs in background thread)."""
    try:
        token = os.environ.get("DISCORD_BOT_TOKEN", "")
        if not token:
            return
        channel = os.environ.get("MC_DISCORD_CHANNEL", "915789984282325016")
        _discord_session.post(
            f"https://discord.com/api/v10/channels/{channel}/messages",
            headers={"Authorization": f"Bot {token}"},
            data={"content": f"**[{agent_name}]** tick {tick}: {action_text}"},
            files={"file": ("screenshot.png", io.BytesIO(png_bytes), "image/png")},
            timeout=10,
        )
    except Exception as e:
        print(f"  Discord post error: {e}")

def post_screenshot_to_discord(img, agent_name, action_text, tick):
    """Post screenshot to Discord in background (non-blocking)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    _io_pool.submit(_post_screenshot_sync, buf.getvalue(), agent_name, action_text, tick)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    global _llm_future
    parser = ArgumentParser(description="DiscoClaw Craftium Orchestrator")
    parser.add_argument("--agents", type=int, default=4, help="Number of agents (1-4)")
    parser.add_argument("--goal", type=str, default="Gather wood, then build an awesome house! Chop trees first, then place blocks to build walls and a roof.",
                        help="Initial goal for the crew")
    parser.add_argument("--screenshots", type=str, default=None, help="Save screenshots to this dir")
    parser.add_argument("--discord-interval", type=int, default=30, help="Post to Discord every N ticks")
    parser.add_argument("--max-ticks", type=int, default=100000, help="Max game ticks")
    parser.add_argument("--obs-size", type=int, default=256, help="Observation image size")
    parser.add_argument("--frameskip", type=int, default=8, help="Frames to skip between agent decisions")
    parser.add_argument("--xvfb-display", type=str, default=None,
                        help="Render game to this X display (e.g. host.docker.internal:98) for x11grab streaming")
    parser.add_argument("--llm-every", type=int, default=10,
                        help="Call LLM every N ticks; repeat last action between calls")
    parser.add_argument("--game", type=str, default=None,
                        help="Luanti game ID to load (e.g. backroomtest, capturetheflag, glitch, void). Default: VoxeLibre")
    parser.add_argument("--env", type=str, default="Craftium/OpenWorld-v0",
                        help="Gymnasium env ID (Craftium/OpenWorld-v0, Craftium/ProcDungeons-v0, Craftium/Speleo-v0, etc.)")
    args = parser.parse_args()

    num_agents = min(args.agents, 4)
    active_agents = AGENTS[:num_agents]
    goal = args.goal

    if args.screenshots:
        os.makedirs(args.screenshots, exist_ok=True)

    print(f"=== DiscoClaw Craftium ===")
    print(f"Agents: {', '.join(a['name'] for a in active_agents)}")
    print(f"Goal: {goal}")
    print(f"Game: {args.game or 'VoxeLibre (default)'}")
    print()

    # Create environment
    config = dict(
        max_block_generate_distance=3,
        fov=90,
        console_alpha=0,
        smooth_lighting=False,
        performance_tradeoffs=True,
        enable_particles=False,
        mg_name="valleys",  # valleys = mountains/rivers/forests, flat = backrooms
        time_speed=0,  # freeze time (no day/night cycle)
        static_spawntime=5500,  # permanent morning light
        # Peaceful — no hostile mobs (zombies, skeletons, etc.)
        only_peaceful_mobs=True,
        mcl_mobs_spawn=False,  # VoxeLibre mob spawning
        mobs_spawn=False,
    )

    if num_agents == 1:
        # Single agent mode
        # If --xvfb-display given, render to that X display for x11grab streaming
        if args.xvfb_display:
            os.environ["DISPLAY"] = args.xvfb_display
            os.environ["SDL_VIDEODRIVER"] = "x11"
            os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
            print(f"Rendering to X display {args.xvfb_display}")

        # Use 16:9 window for stream, LLM gets downscaled in obs_to_base64
        game_w = 1280 if args.xvfb_display else args.obs_size
        game_h = 720 if args.xvfb_display else args.obs_size
        gym_kwargs = dict(
            frameskip=args.frameskip,
            obs_width=game_w,
            obs_height=game_h,
            minetest_conf=config,
            sync_mode=True,
            pmul=3,  # default is 20 (way too fast). 3 = natural walking speed
            mt_listen_timeout=180_000,  # 3 min — non-VoxeLibre envs load slow
        )
        if args.game:
            gym_kwargs["game_id"] = args.game
        env = gym.make(args.env, **gym_kwargs)

        # Detect action space and build remapping if env has fewer actions
        n_actions = env.action_space.n if hasattr(env.action_space, 'n') else len(ACTION_NAMES)
        # Get the wrapper's action names if available
        env_actions = None
        inner = env
        while hasattr(inner, 'env'):
            if hasattr(inner, 'actions'):
                env_actions = inner.actions
                break
            inner = inner.env
        if env_actions:
            print(f"Env actions ({len(env_actions)}): {env_actions}")
            # Build remap: our ACTION_NAMES index → env discrete action index
            # DiscreteActionWrapper: 0=NOP, 1=first_action, 2=second_action, ...
            # So env_actions[i] corresponds to discrete index i+1
            _remap = {}
            for our_idx, our_name in enumerate(ACTION_NAMES):
                best = None
                if our_name == "do nothing":
                    best = 0  # NOP
                elif "forward" in our_name and "camera" not in our_name:
                    for i, a in enumerate(env_actions):
                        if "forward" in a: best = i + 1; break
                elif "backward" in our_name:
                    for i, a in enumerate(env_actions):
                        if "backward" in a or "back" in a: best = i + 1; break
                elif our_name == "move left":
                    for i, a in enumerate(env_actions):
                        if a == "left": best = i + 1; break
                elif our_name == "move right":
                    for i, a in enumerate(env_actions):
                        if a == "right": best = i + 1; break
                elif "jump" in our_name:
                    for i, a in enumerate(env_actions):
                        if "jump" in a: best = i + 1; break
                elif our_name == "use tool":
                    for i, a in enumerate(env_actions):
                        if "dig" in a: best = i + 1; break
                elif our_name == "sneak":
                    for i, a in enumerate(env_actions):
                        if "sneak" in a: best = i + 1; break
                elif "camera right" in our_name:
                    for i, a in enumerate(env_actions):
                        if "mouse x+" in a or "x+" in a: best = i + 1; break
                elif "camera left" in our_name:
                    for i, a in enumerate(env_actions):
                        if "mouse x-" in a or "x-" in a: best = i + 1; break
                elif our_name == "place":
                    for i, a in enumerate(env_actions):
                        if "place" in a: best = i + 1; break
                elif "hotbar slot" in our_name or "select" in our_name:
                    slot = our_name.split()[-1] if our_name.split() else ""
                    for i, a in enumerate(env_actions):
                        if f"slot_{slot}" in a or f"slot {slot}" in a: best = i + 1; break
                elif "camera up" in our_name:
                    for i, a in enumerate(env_actions):
                        if "mouse y+" in a or "y+" in a: best = i + 1; break
                elif "camera down" in our_name:
                    for i, a in enumerate(env_actions):
                        if "mouse y-" in a or "y-" in a: best = i + 1; break
                if best is not None:
                    _remap[our_idx] = best
                else:
                    _remap[our_idx] = 1 if n_actions > 1 else 0  # fallback: forward
            print(f"Action remap: {_remap}")
        else:
            _remap = {i: i for i in range(len(ACTION_NAMES))}

        def remap_action(idx):
            return _remap.get(idx, 1 if n_actions > 1 else 0)

        # Safe env.step with timeout to prevent infinite hangs
        def _step_timeout_handler(signum, frame):
            raise TimeoutError("env.step() hung for >10s")
        def safe_step(action):
            old = signal.signal(signal.SIGALRM, _step_timeout_handler)
            signal.alarm(10)
            try:
                result = env.step(action)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
            return result

        # Patch proc_env before reset() spawns minetest
        import craftium as _craftium_pkg
        pkg_dir = os.path.dirname(_craftium_pkg.__file__)
        libs_dir = os.path.join(os.path.dirname(pkg_dir), "craftium.libs")
        inner_env = env.unwrapped
        inner_env.mt.proc_env["LD_LIBRARY_PATH"] = libs_dir
        inner_env.mt.proc_env["MESA_GL_VERSION_OVERRIDE"] = "3.3"
        if args.xvfb_display:
            inner_env.mt.proc_env["DISPLAY"] = args.xvfb_display
            inner_env.mt.proc_env["SDL_VIDEODRIVER"] = "x11"

        observation, info = env.reset()
        print(f"[{active_agents[0]['name']}] World loaded. Playing...")

        recent_actions = []
        last_action_idx = 1  # default: move forward
        last_action_name = "move forward"
        last_positions = []  # track position to detect being stuck
        chop_sticky = 0  # when >0, repeat "use tool" for this many more ticks
        horizon_cooldown = 0  # after auto-horizon, skip camera actions for N ticks
        fwd_blocked_count = 0  # consecutive FWD-BLOCKED events

        # Tree chop cycle: after initial chop, cycle look up/down/fwd to get all trunk pieces
        # Sequence: chop x4 → look up → chop x4 → look up → chop x4 → look down x2 → chop x4 → fwd → chop x4
        CHOP_CYCLE = (
            [7]*4 +   # chop at current level
            [16] +     # look up 1 step (camera up = action 16)
            [7]*4 +   # chop upper trunk
            [16] +     # look up 1 more step
            [7]*4 +   # chop even higher
            [17]*2 +   # look back down to eye level
            [7]*3 +   # chop at eye level again
            [1]*2 +   # step forward (into where tree was)
            [7]*3 +   # chop any remaining at new position
            [5] +      # jump
            [7]*2 +   # chop while in air (gets higher pieces)
            [17]*1 +   # look slightly down (undo jump camera drift)
            [1]*2     # move forward to clear the tree area
        )
        chop_cycle_idx = 0  # position in chop cycle
        chop_cycle_active = False  # True when running the cycle
        for tick in range(args.max_ticks):
            agent = active_agents[0]

            # Chop cycle: when active, run through the full tree-chopping sequence
            # (chop, look up, chop, look up, chop, look down, chop, step forward, chop+jump)
            if chop_cycle_active and chop_cycle_idx < len(CHOP_CYCLE):
                action_idx = CHOP_CYCLE[chop_cycle_idx]
                chop_cycle_idx += 1
                action_name = ACTION_NAMES[action_idx]
                recent_actions.append(f"{agent['name']}: {action_name}")
                if len(recent_actions) > 50:
                    recent_actions = recent_actions[-50:]
                img = Image.fromarray(observation)
                observation, reward, terminated, truncated, info = safe_step(remap_action(action_idx))
                pos = info.get("player_pos", [0, 0, 0])
                last_positions.append(pos[:])
                if len(last_positions) > 30:
                    last_positions.pop(0)
                if chop_cycle_idx >= len(CHOP_CYCLE):
                    chop_cycle_active = False
                    chop_cycle_idx = 0
                    # Return to horizon after chop cycle using actual pitch
                    pitch = info.get("player_pitch", 0) if isinstance(info, dict) else 0
                    if abs(pitch) > 20:
                        fix_action = 16 if pitch > 0 else 17  # 16=up, 17=down
                        fix_steps = min(int(abs(pitch) / 25), 3)  # each step ≈ 25°
                        for _ in range(fix_steps):
                            observation, reward, terminated, truncated, info = safe_step(remap_action(fix_action))
                        new_p = info.get("player_pitch", 0) if isinstance(info, dict) else 0
                        print(f"  [{agent['name']}] tick {tick}: CHOP CYCLE complete, horizon fix {pitch:.0f}°→{new_p:.0f}°", flush=True)
                    else:
                        print(f"  [{agent['name']}] tick {tick}: CHOP CYCLE complete (pitch={pitch:.0f}° OK)", flush=True)
                if terminated or truncated:
                    observation, info = env.reset()
                    chop_cycle_active = False
                continue

            # Only call LLM every N ticks; repeat last action otherwise
            if tick % args.llm_every == 0:
                # Check for stuck loop BEFORE calling LLM — override if needed
                forced = force_break_loop(recent_actions, tick)
                if forced is not None:
                    action_idx = forced
                    action_name = ACTION_NAMES[action_idx]
                    last_action_idx = action_idx
                    last_action_name = action_name
                    chop_cycle_active = False  # reset so we don't resume chopping same spot
                    chop_cycle_idx = 0
                    fwd_blocked_count = 0
                    print(f"  [{agent['name']}] tick {tick}: LOOP BREAK -> {action_name}", flush=True)
                    recent_actions.append(f"{agent['name']}: {action_name}")
                    img = Image.fromarray(observation)
                else:
                    # Auto-horizon correction using pitch (from game engine)
                    correction_action, correction_steps = camera_needs_correction(info)
                    if correction_action is not None:
                        pitch_val = info.get("player_pitch", 0) if isinstance(info, dict) else 0
                        label = "UP(16)" if correction_action == 16 else "DOWN(17)"
                        print(f"  [{agent['name']}] tick {tick}: AUTO-HORIZON pitch={pitch_val:.0f}° → action {label} x{correction_steps}", flush=True)
                        for _ in range(correction_steps):
                            observation, reward, terminated, truncated, info = safe_step(remap_action(correction_action))
                        new_pitch = info.get("player_pitch", 0) if isinstance(info, dict) else 0
                        print(f"  [{agent['name']}] tick {tick}: pitch after correction: {new_pitch:.0f}°", flush=True)
                        cam_label = "up" if correction_action == 16 else "down"
                        recent_actions.append(f"{agent['name']}: move camera {cam_label}")

                    img_b64, img = obs_to_base64(observation, max_size=360)
                    # Thinker: smarter model gets higher-res view (720px) for better analysis
                    thinker_b64, _ = obs_to_base64(observation, max_size=720)
                    call_thinker(thinker_b64, info, recent_actions, tick, goal)
                    system, user = build_prompt(agent, info, goal, recent_actions, tick)
                    t0 = time.time()
                    # Use prefetched result if available, otherwise call synchronously
                    if _llm_future is not None and _llm_future.done():
                        try:
                            response = _llm_future.result(timeout=0.1)
                            _llm_future = None
                            dt = time.time() - t0
                            # Prefetch was from previous obs — still useful for action continuity
                        except Exception:
                            response = call_llm_safe(agent, system, user, img_b64)
                            dt = time.time() - t0
                    else:
                        response = call_llm_safe(agent, system, user, img_b64)
                        dt = time.time() - t0
                    action_idx = parse_action(response)
                    # MOVEMENT BLOCKED → CHOP override:
                    # If LLM chose move forward but position hasn't changed since last LLM call,
                    # something is in the way — switch to use tool to mine through it.
                    if action_idx == 1 and len(last_positions) >= args.llm_every:
                        prev_pos = last_positions[-(args.llm_every)]
                        cur_est = last_positions[-1]
                        dx = abs(cur_est[0] - prev_pos[0])
                        dz = abs(cur_est[2] - prev_pos[2])
                        if dx < 0.5 and dz < 0.5:
                            fwd_blocked_count += 1
                            if fwd_blocked_count >= 4:
                                # Stuck too long mining — go around instead
                                action_idx = random.choice([3, 4, 14, 15])  # strafe or turn
                                fwd_blocked_count = 0
                                print(f"  [{agent['name']}] tick {tick}: FWD-BLOCKED x4 → go around: {ACTION_NAMES[action_idx]}", flush=True)
                            else:
                                action_idx = 7  # use tool
                                print(f"  [{agent['name']}] tick {tick}: FWD-BLOCKED → use tool ({fwd_blocked_count}/4)", flush=True)
                        else:
                            fwd_blocked_count = 0
                    # SAFETY: if Y is dropping, ban digging to prevent deeper holes
                    cur_y = info.get("player_pos", [0,0,0])[1]
                    if cur_y < 10 and action_idx == 7:  # use tool at low Y (below surface)
                        action_idx = random.choice([1, 5, 14, 15])  # forward, jump, camera
                        print(f"  [{agent['name']}] tick {tick}: BLOCKED dig at Y={cur_y:.0f}, forced {ACTION_NAMES[action_idx]}", flush=True)
                    # HARD RULE: if last action was ANY camera move, force movement (no consecutive camera)
                    recent_names = [a.split(": ",1)[-1] for a in recent_actions[-4:]]
                    if len(recent_names) >= 1 and recent_names[-1] in CAMERA_ACTIONS and action_idx in (14, 15, 16, 17):
                        action_idx = 1  # force move forward
                        print(f"  [{agent['name']}] tick {tick}: BLOCKED consecutive camera, forced FORWARD", flush=True)
                    action_name = ACTION_NAMES[action_idx]
                    # Chop cycle: if "use tool", activate full tree-chop sequence
                    if action_idx == 7 and not chop_cycle_active:
                        chop_cycle_active = True
                        chop_cycle_idx = 0
                        print(f"  [{agent['name']}] tick {tick}: CHOP CYCLE started", flush=True)
                    last_action_idx = action_idx
                    last_action_name = action_name
                    print(f"  [{agent['name']}] tick {tick}: {response} -> {action_name} ({dt:.1f}s)", flush=True)
                    recent_actions.append(f"{agent['name']}: {action_name}")
                    if len(recent_actions) > 50:
                        recent_actions = recent_actions[-50:]
                    # Write thinking to file for stream overlay
                    try:
                        thinking = {"agent": agent["name"], "tick": tick, "action": action_name}
                        for line in response.split("\n"):
                            l = line.strip()
                            if l.upper().startswith("OBS:"): thinking["obs"] = l.split(":",1)[1].strip()[:120]
                            elif l.upper().startswith("GOAL:"): thinking["goal"] = l.split(":",1)[1].strip()[:80]
                        with open("/tmp/ai-thinking.json", "w") as f:
                            json.dump(thinking, f)
                    except:
                        pass
            else:
                action_idx = last_action_idx
                action_name = last_action_name
                img = Image.fromarray(observation)

            # Jump = jump + forward
            if action_name == "jump":
                safe_step(remap_action(action_idx))
                observation, reward, terminated, truncated, info = safe_step(remap_action(1))
            else:
                observation, reward, terminated, truncated, info = safe_step(remap_action(action_idx))

            # Fire off LLM prefetch with NEW observation (ready for next LLM tick)
            if tick % args.llm_every == 0 and not chop_cycle_active:
                next_b64, _ = obs_to_base64(observation, max_size=360)
                next_sys, next_usr = build_prompt(agent, info, goal, recent_actions, tick)
                _llm_future = _llm_pool.submit(call_llm_safe, agent, next_sys, next_usr, next_b64)

            # Track position — detect physically stuck
            pos = info.get("player_pos", [0, 0, 0])
            last_positions.append(pos[:])
            if len(last_positions) > 30:
                last_positions.pop(0)
            if len(last_positions) >= 20 and tick % args.llm_every == 0 and not chop_cycle_active:
                old_pos = last_positions[-20]
                dx = abs(pos[0] - old_pos[0])
                dz = abs(pos[2] - old_pos[2])
                if dx < 1 and dz < 1:
                    # Don't escape if we've been actively chopping — position unchanged while
                    # mining is normal. Only escape if not making progress AND not chopping.
                    recent_names_20 = [a.split(": ")[-1] if ": " in a else a for a in recent_actions[-20:]]
                    chop_count = sum(1 for a in recent_names_20 if a == "use tool")
                    if chop_count >= 2:
                        # Been chopping — let it continue, don't escape
                        pass
                    else:
                        print(f"  [STUCK] Position unchanged 20 ticks at Y={pos[1]:.0f} (chops={chop_count})", flush=True)
                        # ESCAPE: jump + strafe — more effective against tree walls than walk
                        direction = random.choice([3, 4])  # left or right strafe
                        print(f"  [ESCAPE] Jump + strafe", flush=True)
                        safe_step(remap_action(5))  # jump
                        for _ in range(3):
                            safe_step(remap_action(direction))  # strafe
                        for _ in range(2):
                            observation, reward, terminated, truncated, info = safe_step(remap_action(1))  # forward
                        new_y = info.get("player_pos", [0, 0, 0])[1]
                        print(f"  [ESCAPE] After escape: Y={new_y:.0f} (was {pos[1]:.0f})", flush=True)
                        last_positions.clear()

            # Screenshots — save every tick so the stream has fresh frames
            if args.screenshots:
                img.save(os.path.join(args.screenshots, f"{agent['name']}_{tick:05d}.png"))

            if tick % args.discord_interval == 0 and tick > 0:
                post_screenshot_to_discord(img, agent['name'], action_name, tick)

            # Cost log every 50 ticks
            if tick > 0 and tick % 50 == 0:
                elapsed_h = (time.time() - _cost_data["start_time"]) / 3600
                total = sum(v.get("cost_usd", 0) for v in _cost_data.values() if isinstance(v, dict))
                print(f"  [COST] tick {tick}: ${total:.4f} total ({elapsed_h:.2f}h, ${total/max(elapsed_h,0.001):.2f}/hr)", flush=True)

            # Learn from experience — store action patterns every 100 ticks
            if tick > 0 and tick % 100 == 0 and len(recent_actions) >= 10:
                last_10 = recent_actions[-10:]
                action_counts = {}
                for a in last_10:
                    act = a.split(": ")[-1] if ": " in a else a
                    action_counts[act] = action_counts.get(act, 0) + 1
                pos = info.get("player_pos", [0, 0, 0])
                summary = (
                    f"Tick {tick}, pos ({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f}). "
                    f"Last 10 actions: {action_counts}. "
                    f"Goal: {goal}."
                )
                store_memory(summary)
                print(f"  [memory] stored experience at tick {tick}", flush=True)

            if terminated or truncated:
                store_memory(f"Episode ended at tick {tick}. Goal was: {goal}. Learned: need to vary actions more and look for resources.")
                print(f"  Episode ended at tick {tick}")
                observation, info = env.reset()

        # End of session learning
        store_memory(f"Completed {args.max_ticks} ticks. Actions taken: {len(recent_actions)}. Goal: {goal}.")
        env.close()

    else:
        # Multi-agent mode
        from craftium import MarlCraftiumEnv

        # Find env directory
        env_dir = None
        for candidate in [
            "/opt/craftium-envs/voxel-libre2",
            "/tmp/craftium/craftium-envs/voxel-libre2",
            os.path.expanduser("~/craftium/craftium-envs/voxel-libre2"),
        ]:
            if os.path.isdir(candidate):
                env_dir = candidate
                break

        if not env_dir:
            print("ERROR: Could not find voxel-libre2 env directory")
            sys.exit(1)

        print(f"Using env: {env_dir}")

        # MarlCraftiumEnv uses mt_server_conf, and needs minetest_dir pointing
        # to the luanti/ subdir where bin/minetest lives
        import craftium as _craftium_pkg
        pkg_dir = os.path.dirname(_craftium_pkg.__file__)
        luanti_dir = os.path.join(pkg_dir, "luanti")

        # The minetest binary has an rpath to craftium.libs/ that breaks when
        # copied to a run dir. We must inject LD_LIBRARY_PATH into proc_env
        # since Popen(env=proc_env) replaces the full environment.
        libs_dir = os.path.join(os.path.dirname(pkg_dir), "craftium.libs")

        env = MarlCraftiumEnv(
            num_agents=num_agents,
            env_dir=env_dir,
            obs_width=args.obs_size,
            obs_height=args.obs_size,
            sync_mode=True,
            frameskip=args.frameskip,
            game_id="VoxeLibre",
            minetest_dir=luanti_dir,
            mt_server_conf=config,
            init_frames=400,
            mt_listen_timeout=180000,
        )

        # Patch proc_env on server + all clients to include craftium.libs
        env.mt_server.proc_env["LD_LIBRARY_PATH"] = libs_dir
        env.mt_server.proc_env["MESA_GL_VERSION_OVERRIDE"] = "3.3"
        for client in env.mt_clients:
            if client.proc_env is None:
                client.proc_env = {}
            client.proc_env["LD_LIBRARY_PATH"] = libs_dir
            client.proc_env["SDL_VIDEODRIVER"] = "offscreen"
            client.proc_env["MESA_GL_VERSION_OVERRIDE"] = "3.3"

        observations, infos = env.reset()
        print(f"World loaded with {num_agents} agents. Playing...")

        recent_actions = []
        for tick in range(args.max_ticks):
            for agent_id, agent in enumerate(active_agents):
                obs = observations[agent_id]
                # MarlCraftiumEnv._get_info returns empty dict, fake position
                info = {"player_pos": [0, 0, 0]}

                img_b64, img = obs_to_base64(obs)
                system, user = build_prompt(agent, info, goal, recent_actions, tick)

                t0 = time.time()
                response = call_llm(agent, system, user, img_b64)
                dt = time.time() - t0

                action_idx = parse_action(response)
                action_name = ACTION_NAMES[action_idx]
                action_dict = action_idx_to_dict(action_idx)

                print(f"  [{agent['name']:8s}] tick {tick:3d}: {response:30s} -> {action_name:20s} ({dt:.1f}s)")
                sys.stdout.flush()
                recent_actions.append(f"{agent['name']}: {action_name}")

                obs_new, reward, terminated, truncated, info_new = env.step_agent(action_dict)
                observations[agent_id] = obs_new

                # Screenshots — save every tick so the stream has fresh frames
                if args.screenshots:
                    img.save(os.path.join(args.screenshots, f"{agent['name']}_{tick:05d}.png"))

                if tick % args.discord_interval == 0 and tick > 0 and agent_id == 0:
                    post_screenshot_to_discord(img, agent['name'], action_name, tick)

            if tick % 50 == 0:
                print(f"  --- tick {tick}/{args.max_ticks} ---")

        env.close()

    # Print cost summary
    elapsed_h = (time.time() - _cost_data["start_time"]) / 3600
    total = sum(v.get("cost_usd", 0) for v in _cost_data.values() if isinstance(v, dict))
    print(f"\n=== Session complete ===")
    print(f"Duration: {elapsed_h:.2f}h | Total cost: ${total:.4f} (${total/max(elapsed_h,0.001):.4f}/hr)")
    for provider in ["gemini", "anthropic"]:
        p = _cost_data.get(provider, {})
        if p.get("calls", 0) > 0:
            print(f"  {provider}: {p['calls']} calls, {p['input_tokens']} in / {p['output_tokens']} out, ${p['cost_usd']:.4f}")
            if provider == "anthropic" and p.get("cache_read_tokens", 0) > 0:
                print(f"    cache: {p['cache_read_tokens']} read / {p['cache_write_tokens']} write tokens")


if __name__ == "__main__":
    main()
