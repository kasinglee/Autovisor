"""Render reproducible test diagrams with known targets, without a real website."""
import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "tests/fixtures/vision"

async def main():
    cases = json.loads((DEST / "cases.json").read_text())
    cases = [case for case in cases if not case["id"].startswith("synthetic-")]
    scenes = [
        ("synthetic-green-F", "Click the uppercase F with the same color as S", "请点击大写S颜色一样的大写F",
         '<text x="32" y="78" fill="#56bddf">F</text><text x="126" y="80" fill="#72ba54">S</text><text x="54" y="144" fill="#72ba54">F</text><text x="228" y="144" fill="#e4b94f">T</text>', [155,560,300,930]),
        ("synthetic-blue-cube", "Click the blue cube", "请点击蓝色立方体",
         '<rect x="30" y="30" width="55" height="55" fill="#cf655e"/><rect x="208" y="86" width="55" height="55" fill="#4e83ee"/><rect x="126" y="39" width="55" height="55" fill="#71b754"/>', [650,538,822,882]),
    ]
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True)
        try:
            page = await browser.new_page(viewport={"width":352,"height":298}, device_scale_factor=1)
            for name, english, chinese, shapes, box in scenes:
                html = f'''<style>body{{margin:0;font:14px sans-serif}} header{{height:51px;padding:16px;box-sizing:border-box;border-bottom:1px solid #ddd}} svg{{position:absolute;left:16px;top:67px;background:#e7e7e7}} footer{{position:absolute;left:16px;top:242px;width:320px;height:40px;background:#f7f8fa;text-align:center;line-height:40px;font-size:12px}} text{{font: bold 60px Arial}}</style><header>本地生成的视觉定位测试图</header><svg width="320" height="160">{shapes}</svg><footer>{chinese}</footer>'''
                await page.set_content(html)
                await page.screenshot(path=str(DEST / f"{name}.png"))
                cases.append({"id":name,"file":f"{name}.png","crop":[16,67,336,227],"instruction_en":english,"instruction_zh":chinese,"expected_box_1000":box,"synthetic":True})
        finally:
            await browser.close()
    (DEST / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    print("Generated two local diagrams; retained three real screenshots.")

if __name__ == "__main__":
    asyncio.run(main())
