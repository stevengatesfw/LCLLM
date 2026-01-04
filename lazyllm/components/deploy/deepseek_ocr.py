import os
import json
from typing import Optional, Any, Dict

import base64
import os
import tempfile
import lazyllm
from lazyllm import launchers

from .base import LazyLLMDeployBase
from ..utils.file_operate import _base64_to_file, infer_file_extension


class _DeepSeekOCR(object):
    """DeepSeek-OCR 推理封装。

    该模型是一个 VLM/LLM 风格的 OCR（非 PaddleOCR det/rec 结构），需要使用 transformers
    加载并调用 `model.infer(...)`。
    """

    def __init__(
        self,
        model_name_or_path: str = "deepseek-ai/DeepSeek-OCR",
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown format.",
        max_tokens: int = 16384,
        temperature: float = 0.0,
        # DeepSeek-OCR vLLM 推荐的 ngram logits processor 参数（可按需调整）
        ngram_size: int = 30,
        window_size: int = 90,
        whitelist_token_ids: Optional[list] = None,
        **kw,
    ):
        self.model_name_or_path = model_name_or_path
        self.prompt = prompt
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.ngram_size = int(ngram_size)
        self.window_size = int(window_size)
        self.whitelist_token_ids = whitelist_token_ids or [128821, 128822]

        self.init_flag = lazyllm.once_flag()

    def _load_model(self):
        # 使用 vLLM 官方支持的 DeepSeek-OCR 多模态推理接口，避免 transformers 版本冲突
        try:
            from vllm import LLM
            from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
        except ImportError as e:
            lazyllm.LOG.error(f'DeepSeekOCR: Failed to import vLLM or NGramPerReqLogitsProcessor: {e}')
            raise RuntimeError(f'DeepSeekOCR requires vLLM with DeepSeek-OCR support. Import error: {e}') from e

        self._logits_processors = [
            NGramPerReqLogitsProcessor,
        ]

        # 允许通过环境变量做一些开关（和官方示例一致）
        enable_prefix_caching = os.getenv("DEEPSEEK_OCR_ENABLE_PREFIX_CACHING", "false").lower() == "true"
        mm_processor_cache_gb = float(os.getenv("DEEPSEEK_OCR_MM_CACHE_GB", "0"))
        gpu_mem_util = float(os.getenv("DEEPSEEK_OCR_GPU_MEMORY_UTILIZATION", "0.5"))
        max_model_len = int(os.getenv("DEEPSEEK_OCR_MAX_MODEL_LEN", "4096"))

        lazyllm.LOG.info(f'DeepSeekOCR: Loading model from {self.model_name_or_path}')
        try:
            self._llm = LLM(
                model=self.model_name_or_path,
                enable_prefix_caching=enable_prefix_caching,
                mm_processor_cache_gb=mm_processor_cache_gb,
                logits_processors=[NGramPerReqLogitsProcessor],
                gpu_memory_utilization=gpu_mem_util,
                max_model_len=max_model_len,
                # 当前环境 torch==2.9.0 时，vLLM 的 torch.compile 路径会触发 torch._dynamo 不支持的 builtin，
                # 导致 EngineCore 初始化失败；强制 eager 可稳定启动。
                enforce_eager=True,
            )
            lazyllm.LOG.info('DeepSeekOCR: Model loaded successfully')
        except Exception as e:
            lazyllm.LOG.error(f'DeepSeekOCR: Failed to load model: {e}', exc_info=True)
            raise

    def __call__(self, input_data):
        lazyllm.call_once(self.init_flag, self._load_model)

        # 兼容模型测试工作流传入的 JSON 字符串：
        # 例如: '{"query":"xxx","files":["/path/to/img.jpg"]}'
        if isinstance(input_data, str):
            s = input_data.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    input_data = json.loads(s)
                except Exception:
                    # 保持原样，按原逻辑兜底
                    pass

        # 输入：支持 {'inputs': [...]}，与现有 OCRDeploy 一致
        # 兼容前端“模型测试”会传 files（上传接口返回 file_path）
        if isinstance(input_data, dict):
            file_list = (
                input_data.get("inputs")
                or input_data.get("ocr_files")
                or input_data.get("files")  # 兼容 {'files': [...]}
            )
            if isinstance(file_list, str):
                file_list = [file_list]
        else:
            file_list = input_data
            if isinstance(file_list, str):
                file_list = [file_list]

        if not file_list:
            raise ValueError("DeepSeekOCRDeploy: inputs is required")

        # files 可能是 [{id,value,type}] 结构，抽取 value
        extracted = []
        for it in file_list:
            if isinstance(it, dict) and "value" in it:
                extracted.append(it.get("value"))
            else:
                extracted.append(it)
        file_list = extracted

        # 兼容两种输入：
        # 1) dataURL(base64 + mime)，走 _base64_to_file
        # 2) 纯 base64（无 mime），这里自行 decode 后按 magic 推断后缀落盘
        normalized_files = []
        for file in file_list:
            # 允许直接传文件路径（上传接口返回的 file_path）
            if isinstance(file, str) and (file.startswith("/") or file.startswith("./")):
                normalized_files.append(file)
                continue
            if isinstance(file, str) and file.startswith("data:"):
                normalized_files.append(_base64_to_file(file))
                continue
            # 纯 base64：decode -> bytes_to_file
            try:
                raw = base64.b64decode(file)
            except Exception:
                # 不是 base64，尝试按路径处理（_base64_to_file 内部也不支持）
                normalized_files.append(file)
                continue
            suffix = infer_file_extension(raw)
            tmpdir = lazyllm.config["temp_dir"]
            os.makedirs(tmpdir, exist_ok=True)
            fp = tempfile.NamedTemporaryFile(
                prefix="raw_base64_",
                suffix=suffix,
                dir=tmpdir,
                delete=False,
            ).name
            with open(fp, "wb") as f:
                f.write(raw)
            os.chmod(fp, 0o644)
            normalized_files.append(fp)

        file_list = normalized_files
        lazyllm.LOG.info(f"deepseek-ocr read files: {file_list}")

        from vllm import SamplingParams
        from PIL import Image
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

        sampling_param = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_args=dict(
                ngram_size=self.ngram_size,
                window_size=self.window_size,
                whitelist_token_ids=set(self.whitelist_token_ids),
            ),
            skip_special_tokens=False,
        )

        # batch 推理
        model_input = []
        for file_path in file_list:
            image = Image.open(file_path).convert("RGB")
            
            # 预处理图片：如果图片太大，按比例缩放而不是裁剪，避免识别不完整
            # DeepSeek-OCR 推荐的最大尺寸：base_size=1024 或 1280
            # 为了确保完整识别，我们使用 1024 作为最大尺寸，保持宽高比
            max_size = 1024
            width, height = image.size
            if width > max_size or height > max_size:
                # 按比例缩放，保持宽高比
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                lazyllm.LOG.info(f"Resized image from {width}x{height} to {new_width}x{new_height} to avoid cropping")
            
            model_input.append(
                {"prompt": self.prompt, "multi_modal_data": {"image": image}}
            )

        outputs = self._llm.generate(model_input, sampling_param)
        results = []
        for idx, out in enumerate(outputs):
            try:
                text = out.outputs[0].text
            except Exception:
                text = str(out)
            
            # 按文件分隔显示，保留 markdown 格式
            if len(file_list) > 1:
                file_name = os.path.basename(file_list[idx]) if idx < len(file_list) else f"文件{idx + 1}"
                results.append(f"## {file_name}\n\n{text}")
            else:
                results.append(text)
        
        # 多个文件时用分隔符分隔，单个文件直接返回
        if len(results) > 1:
            return "\n\n---\n\n".join(results)
        else:
            return results[0] if results else ""

    @classmethod
    def rebuild(cls, *args, **kw):
        return cls(*args, **kw)

    def __reduce__(self):
        return _DeepSeekOCR.rebuild, (
            self.model_name_or_path,
            self.prompt,
            self.max_tokens,
            self.temperature,
            self.ngram_size,
            self.window_size,
            self.whitelist_token_ids,
        )


class DeepSeekOCRDeploy(LazyLLMDeployBase):
    """DeepSeek-OCR 部署框架（RelayServer）。

    - **endpoint**: /generate
    - **inputs**: {"inputs": [<base64_file_or_path>]}  (支持单文件/多文件)
    """

    keys_name_handle = {
        "inputs": "inputs",
        "ocr_files": "inputs",
        "files": "inputs",
    }
    message_format = {"inputs": "/path/to/image_or_pdf"}
    default_headers = {"Content-Type": "application/json"}

    def __init__(
        self,
        launcher=launchers.remote(ngpus=1),
        log_path=None,
        trust_remote_code=True,
        port=None,
        prompt: Optional[str] = None,
        max_tokens: int = 16384,
        temperature: float = 0.0,
        ngram_size: int = 30,
        window_size: int = 90,
        whitelist_token_ids: Optional[list] = None,
        **kw,
    ):
        super().__init__(launcher=launcher)
        self._log_path = log_path
        self._trust_remote_code = trust_remote_code
        self._port = port

        self._prompt = prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._ngram_size = ngram_size
        self._window_size = window_size
        self._whitelist_token_ids = whitelist_token_ids

    def __call__(self, finetuned_model=None, base_model=None):
        if not finetuned_model:
            finetuned_model = base_model
        if not finetuned_model:
            finetuned_model = "deepseek-ai/DeepSeek-OCR"

        prompt = (
            self._prompt
            or "<image>\n<|grounding|>Convert the document to markdown format."
        )

        return lazyllm.deploy.RelayServer(
            port=self._port,
            func=_DeepSeekOCR(
                finetuned_model,
                prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                ngram_size=self._ngram_size,
                window_size=self._window_size,
                whitelist_token_ids=self._whitelist_token_ids,
            ),
            launcher=self._launcher,
            log_path=self._log_path,
            cls="deepseek-ocr",
        )()


