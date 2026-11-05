"""Captura del frontend con Chrome headless vía DevTools Protocol (tiempo real, sin reloj virtual).
Uso: python carga/captura_frontend.py "<url>" figuras/frontend.png
"""
import asyncio, base64, json, subprocess, sys, tempfile, time
import aiohttp

CHROME = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
PUERTO = 9333


async def main(url, salida, ancho=1400, alto=1250):
    perfil = tempfile.mkdtemp()
    proc = subprocess.Popen([CHROME, '--headless=new', f'--remote-debugging-port={PUERTO}', f'--user-data-dir={perfil}',
                             '--hide-scrollbars', f'--window-size={ancho},{alto}', 'about:blank'])
    try:
        async with aiohttp.ClientSession() as s:
            for _ in range(50):
                try:
                    tabs = await (await s.get(f'http://127.0.0.1:{PUERTO}/json')).json(); break
                except aiohttp.ClientError:
                    await asyncio.sleep(0.2)
            ws_url = next(t for t in tabs if t['type'] == 'page')['webSocketDebuggerUrl']
            async with s.ws_connect(ws_url, max_msg_size=0) as ws:
                n = 0

                async def cdp(metodo, **params):
                    nonlocal n
                    n += 1; await ws.send_json({'id': n, 'method': metodo, 'params': params})
                    async for m in ws:
                        d = json.loads(m.data)
                        if d.get('id') == n:
                            return d.get('result', {})

                await cdp('Emulation.setDeviceMetricsOverride', width=ancho, height=alto, deviceScaleFactor=1, mobile=False)
                await cdp('Page.navigate', url=url)
                await asyncio.sleep(4)                       # la página hace la búsqueda al cargar
                img = await cdp('Page.captureScreenshot', format='png')
                open(salida, 'wb').write(base64.b64decode(img['data']))
                print('captura guardada en', salida)
    finally:
        proc.terminate()


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1], sys.argv[2]))
