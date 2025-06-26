import os, json, datetime
import numpy as np
from PIL import Image, PngImagePlugin
import torch
import folder_paths                          # ComfyUI 自带

class SaveImageCustom:
    """
    Save image to (save_dir / filename).{png|jpg}
    • 支持 (C,H,W) 与 (H,W,C) 两种布局
    • 不附加批次编号/时间戳（多张批量时仅加 _0000…）
    • 无 DPI 相关逻辑
    """

    # -------- 输入参数 --------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":   ("IMAGE",),
                "save_dir": ("STRING", {"default": "./outputs"}),
                "filename": ("STRING", {"default": "output"}),        # 不带扩展名
                "format":   (["png", "jpg"],),
                "quality":  ("INT", {"default": 90, "min": 10, "max": 100, "step": 1}),
                "preview":  ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt":        "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION     = "save_custom"
    OUTPUT_NODE  = True
    CATEGORY     = "🖼SaveUtility"

    # -------- 内部工具：张量 → PIL --------
    def _tensor_to_pil(self, t: torch.Tensor) -> Image.Image:
        if t.ndim != 3:
            raise ValueError(f"期望 3 维张量，但得到 shape={tuple(t.shape)}")

        # 判断通道位置
        if t.shape[0] <= 4:                     # (C,H,W)
            arr = t.permute(1, 2, 0).cpu().numpy() * 255.0
            channels = t.shape[0]
        elif t.shape[-1] <= 4:                  # (H,W,C)
            arr = t.cpu().numpy() * 255.0
            channels = t.shape[-1]
        else:
            raise ValueError(f"通道数异常：shape={tuple(t.shape)}")

        if channels > 4:
            raise ValueError("通道数量 >4，可能是 latent，请先 VAE Decode。")

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    # -------- 主函数 --------
    def save_custom(self, images, save_dir, filename,
                    format, quality, preview,
                    prompt=None, extra_pnginfo=None):

        os.makedirs(save_dir, exist_ok=True)

        # 组合批处理
        batch = images if isinstance(images, torch.Tensor) else torch.stack(
            [img[0] if isinstance(img, list) else img for img in images])

        # 构建元数据（可选）
        metadata = None
        if prompt or extra_pnginfo:
            metadata = PngImagePlugin.PngInfo()
            if prompt:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo:
                for k, v in extra_pnginfo.items():
                    metadata.add_text(k, json.dumps(v))

        results = []
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        for idx, tensor in enumerate(batch):
            pil_img = self._tensor_to_pil(tensor)

            postfix   = f"_{idx:04d}" if len(batch) > 1 else ""
            stem      = f"{filename}{postfix}"
            save_path = os.path.join(save_dir, f"{stem}.{format}")

            if format == "png":
                pil_img.save(save_path, pnginfo=metadata, compress_level=4)
            else:                                    # jpg
                if pil_img.mode == "RGBA":
                    pil_img = pil_img.convert("RGB")
                pil_img.save(save_path, quality=quality)

            # ---------- 预览 ----------
            if preview:
                root_out = folder_paths.get_output_directory()
                if os.path.commonpath([os.path.abspath(save_dir), root_out]) == os.path.abspath(root_out):
                    subfolder = os.path.relpath(save_dir, root_out)
                    results.append({"filename": f"{stem}.{format}",
                                    "subfolder": subfolder,
                                    "type": "output"})
                else:
                    temp_dir = folder_paths.get_temp_directory()
                    os.makedirs(temp_dir, exist_ok=True)
                    preview_name = f"preview_{now}_{idx}.png"
                    pil_img.save(os.path.join(temp_dir, preview_name))
                    results.append({"filename": preview_name,
                                    "subfolder": "",
                                    "type": "temp"})

        return {"ui": {"images": results}}

