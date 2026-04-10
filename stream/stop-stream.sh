#!/bin/bash
# stop-stream.sh — Kill all stream processes cleanly
echo "=== Stopping Luanti AI Stream ==="
pkill ffmpeg 2>/dev/null && echo "  killed ffmpeg" || true
pkill -f "stream-image-worker" 2>/dev/null && echo "  killed image worker" || true
pkill -f "stream-chatbot" 2>/dev/null && echo "  killed chatbot" || true
pkill -f "stream-thinking" 2>/dev/null && echo "  killed thinking" || true
pkill -f "yt-chat-pytchat" 2>/dev/null && echo "  killed pytchat" || true
pkill -f "stream-watchdog" 2>/dev/null && echo "  killed watchdog" || true
pkill -f "chrome.*luanti-overlay" 2>/dev/null && echo "  killed chrome" || true
pkill -f "Xvfb :98" 2>/dev/null && echo "  killed Xvfb :98" || true
docker stop craftium-discoclaw 2>/dev/null && echo "  stopped container" || true
echo "=== All stopped ==="
