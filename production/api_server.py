#!/usr/bin/env python3
"""
REST API and WebSocket server for Morse code decoding.

Usage:
    python api_server.py --checkpoint ../checkpoints/best_model.pt
    
Then:
    POST /decode with audio file
    WebSocket /ws for real-time streaming
    
Requires:
    pip install fastapi uvicorn python-multipart websockets
"""

import sys
import argparse
import asyncio
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("FastAPI not installed. Install with:")
    print("  pip install fastapi uvicorn python-multipart websockets")
    sys.exit(1)

import numpy as np
from decoder import MorseDecoder, StreamingBuffer, TextMerger

# Global decoder (loaded at startup)
decoder: Optional[MorseDecoder] = None

app = FastAPI(
    title="Morse Code Decoder API",
    description="Decode Morse code audio to text",
    version="1.0.0"
)

# Allow CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Simple web UI for testing
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Morse Decoder</title>
    <style>
        body { 
            font-family: 'Courier New', monospace;
            background: #1a1a2e;
            color: #16c79a;
            padding: 40px;
            max-width: 800px;
            margin: 0 auto;
        }
        h1 { color: #e94560; }
        .controls { margin: 20px 0; }
        button {
            background: #e94560;
            color: white;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            font-family: inherit;
            margin-right: 10px;
        }
        button:hover { background: #ff6b6b; }
        button:disabled { background: #555; }
        #output {
            background: #0f0f23;
            padding: 20px;
            min-height: 200px;
            border: 1px solid #16c79a;
            margin-top: 20px;
            font-size: 18px;
            line-height: 1.5;
        }
        #status {
            color: #888;
            margin-top: 10px;
        }
        .file-upload {
            margin: 20px 0;
            padding: 20px;
            border: 2px dashed #16c79a;
        }
    </style>
</head>
<body>
    <h1>🔊 MORSE CODE DECODER</h1>
    
    <div class="controls">
        <button id="startBtn" onclick="startMic()">🎤 Start Microphone</button>
        <button id="stopBtn" onclick="stopMic()" disabled>⏹ Stop</button>
    </div>
    
    <div class="file-upload">
        <input type="file" id="audioFile" accept="audio/*" onchange="uploadFile()">
        <span>Or upload an audio file</span>
    </div>
    
    <div id="output"></div>
    <div id="status">Ready</div>
    
    <script>
        let ws = null;
        let mediaRecorder = null;
        let audioContext = null;
        
        function startMic() {
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('output').textContent = '';
            document.getElementById('status').textContent = 'Connecting...';
            
            ws = new WebSocket('ws://' + window.location.host + '/ws');
            
            ws.onopen = async () => {
                document.getElementById('status').textContent = 'Listening...';
                
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(stream);
                const processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        const data = e.inputBuffer.getChannelData(0);
                        const int16 = new Int16Array(data.length);
                        for (let i = 0; i < data.length; i++) {
                            int16[i] = Math.max(-32768, Math.min(32767, data[i] * 32768));
                        }
                        ws.send(int16.buffer);
                    }
                };
                
                source.connect(processor);
                processor.connect(audioContext.destination);
            };
            
            ws.onmessage = (event) => {
                const output = document.getElementById('output');
                output.textContent += event.data;
            };
            
            ws.onclose = () => {
                document.getElementById('status').textContent = 'Disconnected';
                stopMic();
            };
        }
        
        function stopMic() {
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('status').textContent = 'Stopped';
            
            if (ws) {
                ws.close();
                ws = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
        }
        
        async function uploadFile() {
            const file = document.getElementById('audioFile').files[0];
            if (!file) return;
            
            document.getElementById('status').textContent = 'Uploading...';
            document.getElementById('output').textContent = '';
            
            const formData = new FormData();
            formData.append('audio', file);
            
            try {
                const response = await fetch('/decode', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                document.getElementById('output').textContent = result.text;
                document.getElementById('status').textContent = 'Done';
            } catch (err) {
                document.getElementById('status').textContent = 'Error: ' + err.message;
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the web UI."""
    return HTML_PAGE


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": decoder is not None}


@app.post("/decode")
async def decode_audio(audio: UploadFile = File(...), beam_search: bool = False):
    """
    Decode an uploaded audio file.
    
    Args:
        audio: Audio file (WAV, MP3, etc.)
        beam_search: Use beam search decoding
        
    Returns:
        Decoded text
    """
    if decoder is None:
        return {"error": "Model not loaded"}
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        text = decoder.decode_file(tmp_path, use_beam_search=beam_search)
        return {"text": text, "beam_search": beam_search}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time audio streaming.
    
    Send raw audio bytes (16-bit PCM, 16kHz, mono).
    Receive decoded text as it becomes available.
    """
    await websocket.accept()
    
    if decoder is None:
        await websocket.send_text("Error: Model not loaded")
        await websocket.close()
        return
    
    buffer = StreamingBuffer(window_duration=10.0, overlap_duration=2.0)
    merger = TextMerger()
    
    try:
        while True:
            # Receive audio bytes
            data = await websocket.receive_bytes()
            
            # Convert to float32
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Process through buffer
            for window in buffer.add(audio):
                # Decode
                text = decoder.decode(window)
                
                # Merge and send new text
                new_text = merger.merge(text)
                if new_text.strip():
                    await websocket.send_text(new_text)
                    
    except WebSocketDisconnect:
        # Client disconnected - flush remaining
        final = buffer.flush()
        if final is not None:
            text = decoder.decode(final)
            new_text = merger.merge(text)
            if new_text.strip():
                try:
                    await websocket.send_text(new_text)
                except:
                    pass


def main():
    global decoder
    
    parser = argparse.ArgumentParser(description='Morse Decoder API Server')
    parser.add_argument('--checkpoint', '-c', type=str, 
                       default='../checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'],
                       help='Device for inference')
    
    args = parser.parse_args()
    
    # Find checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        alt_path = Path(__file__).parent / args.checkpoint
        if alt_path.exists():
            checkpoint_path = alt_path
        else:
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
    
    # Load model
    print("Loading model...")
    decoder = MorseDecoder(str(checkpoint_path), device=args.device)
    
    # Run server
    print(f"\nStarting server at http://{args.host}:{args.port}")
    print("Open in browser for web UI")
    print("POST /decode for API")
    print("WebSocket /ws for streaming")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

