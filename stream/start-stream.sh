#!/bin/bash
# start-stream.sh — One-click startup for the full Luanti/Minecraft AI livestream
# Starts ALL required processes in correct order. Safe to re-run (kills old processes first).
#
# Usage: bash ~/.nemoclaw/custom/scripts/start-stream.sh
#
# What this starts:
#   1. Host Xvfb :98 (overlay display)
#   2. Next.js dev server on :3001
#   3. Chrome kiosk on :98 showing overlay page
#   4. Container Xvfb :99 (game display)
#   5. API keys + orchestrator (AI game agent) inside container
#   6. Wait for world load + resize game window to 1280x720
#   7. ffmpeg dual-display compositor → YouTube RTMP
#   8. pytchat YouTube chat poller
#   9. stream-chatbot (AI chat responses on overlay)
#  10. stream-image-worker (!image command handler)
#  11. stream-thinking (AI thinking overlay poller)
#
# Architecture:
#   Container :99 (TCP) → game renders here
#   Host :98 (unix)     → Chrome overlay renders here
#   ffmpeg grabs both, composites with colorkey, streams to YouTube
#   colorkey=0x000000:0.01:0.0 — similarity MUST be 0.01 (not 0.1)
#
# Orchestrator: Gemini 2.5 Flash Lite (main agent) + Claude Sonnet 4.5 (thinker)
# pmul=3 (natural speed), MOUSE_MOV=0.04 (precise camera), frameskip=4

set -e
SCRIPTS="$HOME/.nemoclaw/custom/scripts"
CRAFTIUM="$HOME/.nemoclaw/custom/craftium"
NETIFY="$HOME/netify-dev"
ENV_FILE="$HOME/.nemoclaw_env"
ASSETS="$HOME/.nemoclaw/custom/assets"
MUSIC_PLAYLIST="$ASSETS/luanti-music-playlist.txt"

# Load env
GOOGLE_API_KEY=$(grep "GOOGLE_API_KEY" "$ENV_FILE" | cut -d= -f2)
ANTHROPIC_API_KEY=$(grep "ANTHROPIC_API_KEY" "$ENV_FILE" | cut -d= -f2)
YT_STREAM_KEY=$(grep "YOUTUBE_STREAM_KEY" "$ENV_FILE" | cut -d= -f2)

echo "=== Luanti AI Stream Startup ==="
echo ""

# ── Kill all old stream processes first ─────────────────────────────
echo "[0/11] Cleaning up old processes..."
pkill -f "Xvfb :98" 2>/dev/null || true
pkill -f "chrome.*luanti-overlay" 2>/dev/null || true
pkill ffmpeg 2>/dev/null || true
pkill -f "yt-chat-pytchat" 2>/dev/null || true
pkill -f "stream-chatbot" 2>/dev/null || true
pkill -f "stream-image-worker" 2>/dev/null || true
pkill -f "stream-thinking" 2>/dev/null || true
pkill -f "stream-watchdog" 2>/dev/null || true
sleep 2
echo "  ✓ Old processes cleaned"

# ── Step 1: Host Xvfb :98 (overlay display) ─────────────────────
echo "[1/11] Xvfb :98 (overlay display)..."
rm -f /tmp/.X98-lock
chmod 1777 /tmp/.X11-unix 2>/dev/null || true
Xvfb :98 -screen 0 1280x720x24 -ac &disown
sleep 2
echo "  ✓ Xvfb :98 running"

# ── Step 2: Next.js dev server ───────────────────────────────────
echo "[2/11] Next.js dev server on :3001..."
if ! pgrep -f "next dev" >/dev/null 2>&1; then
    cd "$NETIFY" && npx next dev -p 3001 > /tmp/nextdev.log 2>&1 &disown
    sleep 5
    echo "  ✓ Next.js started"
else
    echo "  ✓ Next.js already running"
fi

# ── Step 3: Chrome kiosk on :98 ─────────────────────────────────
echo "[3/11] Chrome overlay on :98..."
sleep 1
DISPLAY=:98 "$HOME/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome" \
    --no-sandbox --disable-dev-shm-usage \
    --use-gl=angle --use-angle=swiftshader-webgl \
    --enable-unsafe-swiftshader --enable-webgl --ignore-gpu-blocklist --disable-vulkan \
    --window-size=1280,720 --window-position=0,0 \
    --start-fullscreen --kiosk --no-first-run --no-default-browser-check \
    --autoplay-policy=no-user-gesture-required --disable-infobars \
    "http://localhost:3001/luanti-overlay" > /tmp/chrome-overlay.log 2>&1 &disown
sleep 3
echo "  ✓ Chrome overlay running"

# ── Step 4: Container Xvfb :99 ──────────────────────────────────
echo "[4/11] Container Xvfb :99 (game display)..."
# Start container if stopped, restart if running
docker start craftium-discoclaw 2>/dev/null || true
sleep 5
docker exec craftium-discoclaw bash -c "
    rm -f /tmp/.X99-lock
    pkill Xvfb 2>/dev/null || true
    sleep 1
    Xvfb :99 -screen 0 1280x720x24 -ac -listen tcp &disown
    sleep 2
    echo 'Xvfb :99 up'
"
echo "  ✓ Container Xvfb :99 running"

# ── Step 5: Deploy orchestrator + API keys ───────────────────────
echo "[5/11] API keys + orchestrator..."
# Copy orchestrator from permanent location
docker cp "$CRAFTIUM/orchestrator.py" craftium-discoclaw:/opt/craftium/orchestrator.py
# Write API keys (container /tmp is wiped on restart)
docker exec craftium-discoclaw bash -c "echo '$GOOGLE_API_KEY' > /tmp/gemini_key"
docker exec craftium-discoclaw bash -c "echo '$ANTHROPIC_API_KEY' > /tmp/anthropic_key"
# Write initial hint
docker exec craftium-discoclaw bash -c "echo 'Explore the world! Find trees, chop wood, and build a shelter.' > /tmp/ai-hint.txt"
# Clean old run dirs and start orchestrator
docker exec craftium-discoclaw bash -c "
    rm -rf /opt/craftium/minetest-*
    cd /opt/craftium && DISPLAY=:99 PYTHONUNBUFFERED=1 \
    python3 orchestrator.py --agents 1 --obs-size 360 --xvfb-display :99 --frameskip 4 \
    > /tmp/orch.log 2>&1 &
"
echo "  ✓ Orchestrator starting (takes ~60s to load world)"

# ── Step 6: Wait for world + resize window ───────────────────────
echo "[6/11] Waiting for game world to load..."
for i in $(seq 1 30); do
    if docker exec craftium-discoclaw grep -q "World loaded" /tmp/orch.log 2>/dev/null; then
        echo "  ✓ World loaded!"
        break
    fi
    sleep 3
    printf "."
done
echo ""
sleep 2
docker exec craftium-discoclaw bash -c "
    DISPLAY=:99 xdotool search --name '' 2>/dev/null | tail -1 | \
    xargs -I{} bash -c 'DISPLAY=:99 xdotool windowsize {} 1280 720; DISPLAY=:99 xdotool windowmove {} 0 0'
" 2>/dev/null
echo "  ✓ Game window resized to 1280x720"

# ── Step 7: ffmpeg compositor → YouTube ──────────────────────────
echo "[7/11] ffmpeg stream to YouTube..."
sleep 2
# Copy music playlist to /tmp if not there (ffmpeg reads from /tmp)
cp "$MUSIC_PLAYLIST" /tmp/luanti-music-playlist.txt 2>/dev/null || true
/usr/bin/ffmpeg -y \
    -f x11grab -video_size 1280x720 -framerate 30 -i localhost:99 \
    -f x11grab -video_size 1280x720 -framerate 30 -i :98 \
    -stream_loop -1 -f concat -safe 0 -i /tmp/luanti-music-playlist.txt \
    -filter_complex "[1:v]colorkey=0x000000:0.01:0.0[ovr];[0:v][ovr]overlay=0:0[out]" \
    -map "[out]" -map 2:a -filter:a volume=0.35 \
    -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
    -g 60 -keyint_min 60 -b:v 2500k -maxrate 2500k -bufsize 5000k \
    -c:a aac -b:a 128k -ar 44100 \
    -f flv "rtmps://a.rtmp.youtube.com/live2/$YT_STREAM_KEY" \
    > /tmp/ffmpeg-stream.log 2>&1 &disown
sleep 3
echo "  ✓ Streaming to YouTube"

# ── Step 8: pytchat chat poller ──────────────────────────────────
echo "[8/11] YouTube chat poller (pytchat)..."
sleep 1
python3 "$SCRIPTS/youtube/yt-chat-pytchat.py" > /tmp/pytchat.log 2>&1 &disown
echo "  ✓ Chat poller running"

# ── Step 9: Stream chatbot ───────────────────────────────────────
echo "[9/11] Stream chatbot..."
sleep 1
node "$SCRIPTS/stream-chatbot.js" > /tmp/chatbot.log 2>&1 &disown
echo "  ✓ Chatbot running"

# ── Step 10: Image worker ────────────────────────────────────────
echo "[10/11] Image worker..."
sleep 1
node "$SCRIPTS/stream-image-worker.js" > /tmp/image-worker.log 2>&1 &disown
echo "  ✓ Image worker running"

# ── Step 11: Thinking overlay poller ─────────────────────────────
echo "[11/11] AI thinking poller..."
sleep 1
node "$SCRIPTS/stream-thinking.js" > /tmp/thinking.log 2>&1 &disown
echo "  ✓ Thinking poller running"

echo ""
echo "=== Stream is LIVE ==="
echo ""
echo "Verify:"
echo "  tail -5 /tmp/ffmpeg-stream.log    # should show frame count"
echo "  docker exec craftium-discoclaw tail -5 /tmp/orch.log  # game ticks"
echo "  tail -5 /tmp/chatbot.log          # chat responses"
echo "  tail -5 /tmp/image-worker.log     # image gen"
echo ""
echo "Costs:"
echo "  Main agent: Gemini 2.5 Flash Lite (~\$0.04/hr)"
echo "  Thinker:    Claude Sonnet 4.5 (every 30 ticks, ~\$0.50/hr)"
echo ""
echo "Controls:"
echo "  Hint AI:  docker exec craftium-discoclaw bash -c \"echo 'your hint' > /tmp/ai-hint.txt\""
echo "  Kill all: bash $SCRIPTS/stop-stream.sh"
echo ""
echo "Troubleshooting:"
echo "  No video?  → pkill ffmpeg && re-run this script"
echo "  Game stuck? → docker restart craftium-discoclaw && re-run"
echo "  Chat dead?  → pkill -f pytchat && python3 $SCRIPTS/youtube/yt-chat-pytchat.py &"
