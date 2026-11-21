"""Opt-in Windows smoke test using the normal CLI, Edge and its DevTools protocol."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import pytest


@pytest.mark.skipif(os.getenv("FASTGEARS_BROWSER_TEST") != "1", reason="set FASTGEARS_BROWSER_TEST=1 to launch Edge")
def test_enabled_browser_receives_pattern(tmp_path):
    from websockets.sync.client import connect

    edge = shutil.which("msedge") or str(Path(os.environ["ProgramFiles(x86)"]) /
                                        "Microsoft/Edge/Application/msedge.exe")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "DATA_DIR":str(tmp_path), "PHONE_VISION_CHECK":"0"}
    server = subprocess.Popen([sys.executable, "-m", "fastgears", "--synthetic", "--no-llm",
                               "--host","127.0.0.1","--port",str(port)], env=env)
    browser = None
    try:
        for _ in range(100):
            try:
                with urlopen(url+"/api/health", timeout=.3) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(.1)
        profile = tmp_path / "edge"
        browser = subprocess.Popen([edge, "--headless", "--disable-gpu", "--mute-audio",
                                    "--no-first-run", "--disable-extensions", "--remote-debugging-port=0",
                                    f"--user-data-dir={profile}",url],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        endpoint = profile / "DevToolsActivePort"
        for _ in range(100):
            if endpoint.is_file() and endpoint.read_text().splitlines():
                break
            time.sleep(.1)
        debug_port = endpoint.read_text().splitlines()[0]
        with urlopen(f"http://127.0.0.1:{debug_port}/json/list", timeout=3) as response:
            page = next(p for p in json.load(response) if p["type"] == "page")
        with connect(page["webSocketDebuggerUrl"], open_timeout=3) as ws:
            def evaluate(expression):
                ws.send(json.dumps(dict(id=1, method="Runtime.evaluate", params=dict(
                    expression=expression, returnByValue=True, userGesture=True, awaitPromise=True))))
                while True:
                    reply = json.loads(ws.recv(timeout=5))
                    if reply.get("id") == 1:
                        assert "exceptionDetails" not in reply["result"], reply["result"]
                        return reply["result"]["result"].get("value")

            for _ in range(80):
                if evaluate("document.getElementById('conn')?.textContent === 'live'"):
                    break
                time.sleep(.1)
            evaluate("""window.beeps=0; window.spoken=[];
                const original=AudioContext.prototype.createOscillator;
                AudioContext.prototype.createOscillator=function(){beeps++; return original.call(this)};
                speechSynthesis.speak=u=>spoken.push(u.text);
                document.getElementById('enable').onclick()""")
            for _ in range(140):
                if evaluate("document.getElementById('pattern-name').textContent === 'Repeated distraction'"):
                    break
                time.sleep(.2)
            assert evaluate("document.getElementById('alert-level').textContent") == "CAUTION"
            assert evaluate("beeps") == 1
            assert evaluate("spoken") == ["Bring your attention back to the road."]
            assert evaluate("document.getElementById('recognition').textContent.includes('SYNTHETIC')")
            assert evaluate("document.getElementById('timeline').children.length") >= 3
            assert evaluate("""playAlert({id:'urgent-test',severity:'WARNING',message:'Focus now.'});
                playAlert({id:'urgent-test',severity:'WARNING',message:'Duplicate.'});
                playAlert({id:'lower-test',severity:'CAUTION',message:'Lower priority.'});
                beeps === 3 && spoken.length === 2 && spoken[1] === 'Focus now.'""")
            ws.send(json.dumps(dict(id=2, method="Browser.close")))
    finally:
        if browser:
            try:
                browser.wait(timeout=5)
            except subprocess.TimeoutExpired:
                browser.terminate()
                browser.wait(timeout=5)
        server.terminate()
        server.wait(timeout=5)
