#!/bin/bash
# watch.sh — Local browser viewer for Craftium game
# Opens a low-latency HLS stream of the game display in your browser.
#
# Usage:
#   bash watch.sh              # game only (from container :99)
#   bash watch.sh --overlay    # game + overlay composite (requires :98)
#   bash watch.sh --port 8090  # custom port (default 8088)
#
# Then open: http://localhost:8088
#
# Architecture:
#   ffmpeg grabs X display → HLS segments → Python HTTP server → browser
#   Ultra-low latency: 0.5s segments, 2-segment playlist

set -e

PORT=8088
MODE="game"  # game | composite
HLS_DIR="/tmp/hls-local"

while [[ $# -gt 0 ]]; do
  case $1 in
    --overlay|--composite) MODE="composite"; shift ;;
    --port) PORT="$2"; shift 2 ;;
    *) shift ;;
  esac
done

# Clean old HLS segments
rm -rf "$HLS_DIR"
mkdir -p "$HLS_DIR"

# Write the player HTML
cat > "$HLS_DIR/index.html" << 'HTMLEOF'
<!DOCTYPE html>
<html><head>
<title>Craftium Local Viewer</title>
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0a; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; font-family: monospace; color: #aaa; }
  #container { position: relative; width: 100%; max-width: 1280px; aspect-ratio: 16/9; }
  video { width: 100%; height: 100%; background: #111; border-radius: 4px; }
  #status { position: absolute; top: 8px; right: 12px; font-size: 11px; padding: 2px 8px; border-radius: 3px; background: rgba(0,0,0,0.7); }
  .live { color: #f44; }
  .connecting { color: #fa0; }
  #info { margin-top: 12px; font-size: 12px; opacity: 0.5; }
  #controls { margin-top: 8px; display: flex; gap: 12px; }
  #controls button { background: #222; color: #ccc; border: 1px solid #444; padding: 4px 12px; border-radius: 3px; cursor: pointer; font-family: monospace; font-size: 11px; }
  #controls button:hover { background: #333; }
</style>
</head><body>
<div id="container">
  <video id="v" autoplay muted></video>
  <div id="status" class="connecting">connecting...</div>
</div>
<div id="controls">
  <button onclick="toggleMute()">unmute</button>
  <button onclick="goLive()">go live</button>
  <button onclick="toggleFullscreen()">fullscreen</button>
</div>
<div id="info">Craftium Local Viewer &mdash; HLS low-latency</div>
<script>
const v = document.getElementById('v');
const status = document.getElementById('status');
let hls;

function initPlayer() {
  if (Hls.isSupported()) {
    hls = new Hls({
      liveSyncDurationCount: 1,
      liveMaxLatencyDurationCount: 2,
      lowLatencyMode: true,
      backBufferLength: 0,
      maxBufferLength: 2,
      maxMaxBufferLength: 4,
      manifestLoadingTimeOut: 5000,
      manifestLoadingMaxRetry: 100,
      manifestLoadingRetryDelay: 1000,
      levelLoadingTimeOut: 5000,
      levelLoadingMaxRetry: 100,
      levelLoadingRetryDelay: 1000,
    });
    hls.loadSource('/stream.m3u8');
    hls.attachMedia(v);
    hls.on(Hls.Events.MANIFEST_PARSED, () => {
      v.play().catch(() => {});
      status.textContent = 'LIVE';
      status.className = 'live';
    });
    hls.on(Hls.Events.ERROR, (e, data) => {
      if (data.fatal) {
        status.textContent = 'reconnecting...';
        status.className = 'connecting';
        setTimeout(() => { hls.destroy(); initPlayer(); }, 2000);
      }
    });
  } else if (v.canPlayType('application/vnd.apple.mpegurl')) {
    v.src = '/stream.m3u8';
    v.play().catch(() => {});
  }
}

function toggleMute() {
  v.muted = !v.muted;
  event.target.textContent = v.muted ? 'unmute' : 'mute';
}
function goLive() {
  if (hls) { hls.destroy(); initPlayer(); }
  else { v.currentTime = v.duration; }
}
function toggleFullscreen() {
  if (!document.fullscreenElement) document.getElementById('container').requestFullscreen();
  else document.exitFullscreen();
}

// Auto-retry on load
initPlayer();
// If no manifest after 5s, retry
setTimeout(() => {
  if (status.className === 'connecting') { hls?.destroy(); initPlayer(); }
}, 5000);
</script>
</body></html>
HTMLEOF

echo "=== Craftium Local Viewer ==="
echo "Mode: $MODE | Port: $PORT"
echo "URL: http://localhost:$PORT"
echo ""

# Kill old instances
pkill -f "ffmpeg.*hls-local" 2>/dev/null || true
pkill -f "python3.*hls-local" 2>/dev/null || true
sleep 1

# Start HTTP server (serves HLS dir)
cd "$HLS_DIR"
python3 -m http.server "$PORT" --bind 0.0.0.0 > /tmp/hls-server.log 2>&1 &
SERVER_PID=$!
echo "HTTP server: PID $SERVER_PID on :$PORT"

# Start ffmpeg
if [ "$MODE" = "composite" ]; then
  echo "Grabbing: game (:99 via TCP) + overlay (:98)"
  /usr/bin/ffmpeg -y \
    -f x11grab -video_size 1280x720 -framerate 30 -i localhost:99 \
    -f x11grab -video_size 1280x720 -framerate 30 -i :98 \
    -filter_complex "[1:v]colorkey=0x000000:0.01:0.0[ovr];[0:v][ovr]overlay=0:0[out]" \
    -map "[out]" \
    -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p \
    -g 15 -keyint_min 15 \
    -b:v 3000k -maxrate 3000k -bufsize 3000k \
    -f hls -hls_time 0.5 -hls_list_size 3 -hls_flags delete_segments+append_list \
    -hls_segment_filename "$HLS_DIR/seg%04d.ts" \
    "$HLS_DIR/stream.m3u8" \
    > /tmp/hls-ffmpeg.log 2>&1 &
else
  echo "Grabbing: game only (:99 via TCP)"
  /usr/bin/ffmpeg -y \
    -f x11grab -video_size 1280x720 -framerate 30 -i localhost:99 \
    -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p \
    -g 15 -keyint_min 15 \
    -b:v 3000k -maxrate 3000k -bufsize 3000k \
    -f hls -hls_time 0.5 -hls_list_size 3 -hls_flags delete_segments+append_list \
    -hls_segment_filename "$HLS_DIR/seg%04d.ts" \
    "$HLS_DIR/stream.m3u8" \
    > /tmp/hls-ffmpeg.log 2>&1 &
fi
FFMPEG_PID=$!
echo "ffmpeg: PID $FFMPEG_PID"

sleep 2
if kill -0 $FFMPEG_PID 2>/dev/null; then
  echo ""
  echo "=== READY ==="
  echo "Open in browser: http://localhost:$PORT"
  echo ""
  echo "Controls:"
  echo "  Stop: pkill -f 'hls-local'"
  echo "  Logs: tail -f /tmp/hls-ffmpeg.log"
else
  echo ""
  echo "ERROR: ffmpeg exited. Check /tmp/hls-ffmpeg.log"
  tail -5 /tmp/hls-ffmpeg.log
  kill $SERVER_PID 2>/dev/null
fi
