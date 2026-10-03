"""Bounded DeepSeek image-point verification for visible course dialogs."""

import asyncio
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image
from playwright.async_api import TimeoutError


def parse_point(content, width, height):
    text = content.strip()
    # 有的模型即使要求 JSON 仍附带说明文字，只提取首个 JSON 对象。
    data, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
    if data.get("uncertain") is not False:
        raise ValueError("模型无法确定目标")
    point = data.get("center_px")
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError("模型没有返回单个坐标")
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in point):
        raise ValueError("模型返回无效坐标")
    x, y = point
    if not (0 < x < width and 0 < y < height):
        raise ValueError("模型坐标超出截图范围")
    return x, y


def to_page_point(point, size, box):
    return (
        box["x"] + point[0] * box["width"] / size[0],
        box["y"] + point[1] * box["height"] / size[1],
    )


async def api_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    process = await asyncio.create_subprocess_exec(
        "/usr/bin/security", "find-generic-password", "-s",
        "com.autovisor.deepseek", "-w",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError("读取 DeepSeek 钥匙串超时") from None
    if process.returncode:
        raise RuntimeError("没有可用的 DeepSeek Key")
    return stdout.decode().strip()


async def visible_locator(page, selectors):
    for selector in selectors:
        candidates = page.locator(selector)
        for index in range(await candidates.count()):
            candidate = candidates.nth(index)
            if await candidate.is_visible():
                return candidate
    return None


async def attempt_course_verification(page, logger, config=None):
    """Three attempts per visible dialog; failures return to manual waiting."""
    for attempt in range(1, 4):
        dialog = await visible_locator(page, (".yidun_modal", ".yidun_popup"))
        if dialog is None:
            logger.warn("未找到支持的验证弹窗，请手动完成.")
            return False
        prompt = await dialog.inner_text(timeout=2000)
        if "点击" not in prompt or "滑" in prompt:
            logger.warn("当前不是支持的单点点击验证，请手动完成.")
            return False
        image = await visible_locator(page, (
            ".yidun_modal img.yidun_bg-img", ".yidun_popup img.yidun_bg-img",
        ))
        if image is None:
            logger.warn("没有找到验证图像，停止自动点击，请手动完成.")
            return False
        try:
            screenshot = await dialog.screenshot(scale="css", timeout=3000)
            with Image.open(io.BytesIO(screenshot)) as png:
                size = png.size
            fingerprint = hashlib.sha256(screenshot).hexdigest()[:12]
            diagnostic = Path(logger.log_dir) / f"vision-{fingerprint}.png"
            await dialog.screenshot(path=str(diagnostic), scale="css", timeout=3000)
            image_src = await image.get_attribute("src")
            crop = await image.screenshot(scale="css", timeout=3000)
            provider = getattr(config, "vision_provider", "deepseek")
            scale = getattr(config, "vision_image_scale", 3) if provider == "llama" else 3
            with Image.open(io.BytesIO(crop)) as png:
                enlarged = png.resize((png.width * scale, png.height * scale))
                encoded = io.BytesIO()
                enlarged.save(encoded, format="PNG")
            payload = {
                "model": "deepseek-flash",
                "thinking": {"type": "disabled"},
                "max_tokens": 1200,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": (
                        f"题目文字：{prompt}\n附件只包含题目的字母图像，没有标题和底部提示。"
                        "按题目判断需要点击的单个字母或数字。区分大小写、颜色、"
                        "立体朝向；朝向指正面与侧面的方向，不要把所有直立字母都当作相同朝向。"
                        "返回 JSON，字段 instruction、target、center_px、uncertain。"
                        "center_px 为图像内目标可见正面中心的归一化 [x,y]。"
                        "横纵坐标均为0到1000，原点为附件图像左上角，右下角为[1000,1000]。"
                        "不要点击刷新或按钮。无法确定、需要多点或不是点击题时，"
                        "uncertain=true、center_px=null。只输出 JSON。"
                    )},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode(),
                        "detail": "original",
                    }},
                ]}],
            }
            image_candidates = None
            if provider == "llama":
                local_url = getattr(config, "vision_local_url", "http://127.0.0.1:8091").rstrip("/")
                parsed = urlparse(local_url)
                if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username or parsed.password:
                    raise ValueError("Local vision URL must be localhost HTTP")
                endpoint = local_url + "/v1/chat/completions"
                payload["model"] = "local-vision"
                payload.pop("thinking", None)
                from modules.local_vision import point_response_format
                payload["response_format"] = point_response_format()
                payload["messages"][0]["content"][0]["text"] = (
                    f"题目：{prompt}。图片尺寸为{enlarged.width}x{enlarged.height}像素。"
                    "根据题目定位单个目标，区分大小写、颜色和立体朝向。"
                    "输出JSON字段 target、center_px、uncertain。"
                    "center_px为附件图像内目标正面的绝对像素[x,y]，左上角为[0,0]，"
                    f"右下角为[{enlarged.width-1},{enlarged.height-1}]。"
                    "不确定时uncertain=true，不要猜测。"
                )
                if getattr(config, "vision_local_prompt", "baseline") == "grounded":
                    from modules.vision_candidates import extract_candidates, candidate_image, selection_format, grounded_prompt
                    image_candidates = extract_candidates(enlarged)
                    annotated = candidate_image(enlarged, image_candidates, getattr(config, "vision_large_labels", False))
                    buffer = io.BytesIO()
                    annotated.save(buffer, format="PNG")
                    payload["messages"][0]["content"][1]["image_url"]["url"] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
                    payload["messages"][0]["content"][0]["text"] = grounded_prompt(prompt, image_candidates)
                    payload["response_format"] = selection_format()
                headers = {}
            elif provider == "deepseek":
                endpoint = "https://api.deepseek.com/chat/completions"
                headers = {"Authorization": "Bearer " + await api_key()}
            else:
                raise ValueError("Unknown vision provider")
            logger.event("视觉验证请求", 次数=attempt, 图像=fingerprint, 尺寸=str(size))
            async with httpx.AsyncClient(timeout=120 if provider == "llama" else 45, follow_redirects=False, trust_env=provider != "llama") as client:
                response = await client.post(
                    endpoint, headers=headers, json=payload,
                )
            if response.status_code != 200:
                logger.warn(f"视觉模型请求失败 HTTP {response.status_code}，请手动验证.")
                return False
            content = response.json()["choices"][0]["message"]["content"]
            logger.event("视觉验证模型答案", 答案=content[:1800])
            if image_candidates is not None:
                from modules.local_vision import decode_json
                from modules.vision_candidates import selected_point
                data = decode_json(content)
                content = json.dumps({"center_px": selected_point(data, image_candidates), "uncertain": False})
            point_size = (enlarged.width, enlarged.height) if provider == "llama" else (1000, 1000)
            point = parse_point(content, *point_size)
            if not await dialog.is_visible():
                return True
            if image_src != await image.get_attribute("src"):
                logger.warn("识别期间验证图片已变化，丢弃旧坐标.")
                continue
            box = await dialog.bounding_box()
            image_box = await image.bounding_box()
            if not box or not image_box:
                return False
            x, y = to_page_point(point, point_size, image_box)
            if not (
                image_box["x"] + 5 < x < image_box["x"] + image_box["width"] - 5
                and image_box["y"] + 5 < y < image_box["y"] + image_box["height"] - 5
            ):
                raise ValueError("目标未落在验证图片内部")
            # 再检查图片右上角刷新按钮区域，避免误点触发新题。
            if x > image_box["x"] + image_box["width"] - 45 and y < image_box["y"] + 45:
                raise ValueError("模型目标落在刷新按钮区域")
            logger.event("视觉验证点击", 次数=attempt, 模型坐标=str(point), 坐标范围=str(point_size), 页面坐标=str((round(x, 1), round(y, 1))))
            await page.mouse.click(x, y)
            try:
                await dialog.wait_for(state="hidden", timeout=6000)
                logger.event("视觉验证结果", 结果="弹窗已消失", 次数=attempt)
                return True
            except TimeoutError:
                logger.event("视觉验证结果", 结果="未通过", 次数=attempt, 页面提示=(await dialog.inner_text(timeout=2000))[:400])
                await asyncio.sleep(2)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 不输出 HTTP 请求/响应或密钥内容。
            logger.warn(f"视觉验证停止（{type(exc).__name__}），请手动完成.")
            return False
    logger.warn("视觉验证最多三次仍未通过，停止重试，请手动完成.")
    return False
