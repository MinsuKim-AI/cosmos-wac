"""여러 시뮬레이터의 최신 화면을 한 페이지에 모아 보여 주는 웹 서버 (서버에서 실행).

사용 (서버):   python live_server.py <live_dir> [포트=8765]
보기 (노트북): ssh -N -L 8765:localhost:8765 OASYS   를 켜 둔 채 브라우저로 http://localhost:8765
"""

import functools
import http.server
import os
import sys
import time

LIVE_DIR = os.path.abspath(sys.argv[1])
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8765

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>LIBERO 실시간</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:13px sans-serif}
 header{padding:8px 12px}
 #g{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:6px;padding:6px}
 figure{margin:0;background:#000} img{width:100%;display:block} figcaption{padding:3px 6px}
</style></head><body>
<header>LIBERO 시뮬레이터 실시간. 각 칸은 왼쪽 3인칭, 오른쪽 손목.</header>
<div id="g"></div>
<script>
async function tick(){
  const g = document.getElementById('g');
  try {
    const names = await (await fetch('list', {cache:'no-store'})).json();
    const active = new Set(names.map(n => 'f_'+n));
    for (const f of Array.from(g.children)) {
      if (!active.has(f.id)) f.remove();
    }
    for (const n of names){
      let f = document.getElementById('f_'+n);
      if(!f){
        f = document.createElement('figure');
        f.id = 'f_'+n;
        const caption = document.createElement('figcaption');
        caption.textContent = n;
        f.append(document.createElement('img'), caption);
        g.appendChild(f);
      }
      f.querySelector('img').src = n+'.jpg?'+Date.now();
    }
  } finally {
    setTimeout(tick, 1000);
  }
}
tick();
</script></body></html>"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        elif path == "/list":
            cutoff = time.time() - 45
            with os.scandir(LIVE_DIR) as feeds:
                names = sorted(
                    f.name[:-4] for f in feeds
                    if f.name.endswith(".jpg") and not f.name.startswith(".")
                    and f.stat().st_mtime >= cutoff
                )
            body = ("[" + ",".join(f'"{n}"' for n in names) + "]").encode()
            ctype = "application/json"
        else:
            return super().do_GET()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


os.makedirs(LIVE_DIR, exist_ok=True)
# 127.0.0.1에만 열어서 서버 밖에서는 직접 접속할 수 없다 (SSH 터널로만 본다)
http.server.ThreadingHTTPServer(("127.0.0.1", PORT),
                                functools.partial(Handler, directory=LIVE_DIR)).serve_forever()
