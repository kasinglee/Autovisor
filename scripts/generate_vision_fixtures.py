"""Render reproducible test diagrams with known targets, without a real website."""
import asyncio
import argparse
import json
from pathlib import Path
from PIL import Image
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "tests/fixtures/vision"

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--green-f-source", type=Path)
    args = parser.parse_args()
    cases = json.loads((DEST / "cases.json").read_text())
    cases = [case for case in cases if not case["id"].startswith("synthetic-") and case["id"] != "real-green-F"]
    for case in cases:
        case["split"] = "calibration"
    scenes = [
        ("synthetic-green-F", "Click the uppercase F with the same color as S", "请点击大写S颜色一样的大写F",
         '<text x="32" y="78" fill="#56bddf">F</text><text x="126" y="80" fill="#72ba54">S</text><text x="54" y="144" fill="#72ba54">F</text><text x="228" y="144" fill="#e4b94f">T</text>', [155,560,300,930]),
        ("synthetic-blue-cube", "Click the blue cube", "请点击蓝色立方体",
         '<rect x="30" y="30" width="55" height="55" fill="#cf655e"/><rect x="208" y="86" width="55" height="55" fill="#4e83ee"/><rect x="126" y="39" width="55" height="55" fill="#71b754"/>', [650,538,822,882]),
        ("synthetic-red-P", "Click uppercase P with the same color as uppercase R", "请点击大写R颜色一样的大写P",
         '<text x="30" y="80" fill="#e85d65">R</text><text x="126" y="74" fill="#5498dd">P</text><text x="225" y="142" fill="#e85d65">P</text><text x="100" y="149" fill="#66aa55">T</text>', [690,550,850,930]),
        ("synthetic-lowercase-k", "Click the lowercase k", "请点击小写k",
         '<text x="25" y="73" fill="#5498dd">K</text><text x="134" y="144" fill="#df695d">k</text><text x="236" y="94" fill="#66aa55">X</text>', [400,570,560,930]),
        ("synthetic-yellow-square", "Click the yellow square", "请点击黄色正方形",
         '<rect x="25" y="76" width="58" height="58" fill="#e5bc49"/><rect x="220" y="29" width="58" height="58" fill="#63b7d9"/><circle cx="153" cy="53" r="29" fill="#e5bc49"/>', [78,475,260,838]),
        ("synthetic-purple-8", "Click digit 8 with the same color as uppercase H", "请点击大写H颜色一样的数字8",
         '<text x="26" y="75" fill="#9372bf">H</text><text x="219" y="86" fill="#9372bf">8</text><text x="100" y="145" fill="#e5bc49">8</text>', [660,150,825,560]),
    ]
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True)
        try:
            page = await browser.new_page(viewport={"width":352,"height":298}, device_scale_factor=1)
            for name, english, chinese, shapes, box in scenes:
                html = f'''<style>body{{margin:0;font:14px sans-serif}} header{{height:51px;padding:16px;box-sizing:border-box;border-bottom:1px solid #ddd}} svg{{position:absolute;left:16px;top:67px;background:#e7e7e7}} footer{{position:absolute;left:16px;top:242px;width:320px;height:40px;background:#f7f8fa;text-align:center;line-height:40px;font-size:12px}} text{{font: bold 60px Arial}}</style><header>本地生成的视觉定位测试图</header><svg width="320" height="160">{shapes}</svg><footer>{chinese}</footer>'''
                await page.set_content(html)
                await page.screenshot(path=str(DEST / f"{name}.png"))
                cases.append({"id":name,"file":f"{name}.png","crop":[16,67,336,227],"instruction_en":english,"instruction_zh":chinese,"expected_box_1000":box,"synthetic":True,"split":"calibration" if name in {"synthetic-green-F", "synthetic-blue-cube"} else "holdout"})
        finally:
            await browser.close()
    green_f = DEST / "real-green-F.png"
    if args.green_f_source:
        with Image.open(args.green_f_source) as image:
            image.crop((580,312,1020,685)).save(green_f)
    if not green_f.exists():
        raise RuntimeError("Pass --green-f-source once, or use the checked-in real-green-F.png")
    cases.append({"id":"real-green-F", "file":green_f.name, "crop":[20,84,420,284], "instruction_en":"Click uppercase F with the same color as uppercase S", "instruction_zh":"请点击大写S颜色一样的大写F", "expected_box_1000":[140,580,270,910], "split":"holdout"})
    (DEST / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    print(f"Prepared {len(cases)} cases: five calibration and five holdout.")

if __name__ == "__main__":
    asyncio.run(main())
