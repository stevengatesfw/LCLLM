import os
import base64
import uuid
import importlib.util
from lazyllm.thirdparty import PIL, torch, diffusers
from lazyllm.thirdparty import numpy as np
from io import BytesIO

import lazyllm
from lazyllm import LOG, LazyLLMLaunchersBase
from ..base import LazyLLMDeployBase
from lazyllm.components.formatter import encode_query_with_filepaths
from ...utils.downloader import ModelManager
from ...utils.file_operate import _delete_old_files
from typing import Optional


class _StableDiffusion3(object):

    _load_registry = {}
    _call_registry = {}

    @classmethod
    def register_loader(cls, model_type):
        def decorator(loader_func):
            cls._load_registry[model_type] = loader_func
            return loader_func
        return decorator

    @classmethod
    def register_caller(cls, model_type):
        def decorator(caller_func):
            cls._call_registry[model_type] = caller_func
            return caller_func
        return decorator

    def __init__(self, base_sd, source=None, embed_batch_size=30, trust_remote_code=True, save_path=None, init=False, num_gpus=1):
        source = lazyllm.config['model_source'] if not source else source
        self.base_sd = ModelManager(source).download(base_sd) or ''
        self.embed_batch_size = embed_batch_size
        self.trust_remote_code = trust_remote_code
        self.paintor = None
        self.init_flag = lazyllm.once_flag()
        self.save_path = save_path or os.path.join(lazyllm.config['temp_dir'], 'sd3')
        self.num_gpus = max(1, int(num_gpus)) if num_gpus else 1
        if init:
            lazyllm.call_once(self.init_flag, self.load_sd)

    def load_sd(self):
        if importlib.util.find_spec('torch_npu') is not None:
            import torch_npu  # noqa F401
            from torch_npu.contrib import transfer_to_npu  # noqa F401

        lazyllm.LOG.info(f"load_sd: base_sd='{self.base_sd}', registered loaders: {list(self._load_registry.keys())}")
        for model_type, loader in self._load_registry.items():
            if model_type in self.base_sd.lower():
                lazyllm.LOG.info(f"load_sd: Matched model_type '{model_type}' in base_sd, using registered loader")
                loader(self)
                return
        lazyllm.LOG.info(f"load_sd: No registered loader matched, using default StableDiffusion3Pipeline")

        # 多 GPU 支持：使用 enable_model_cpu_offload 自动分配
        # TODO: 目前只是显存分布式，不是计算分布式，待优化为真正的并行计算
        try:
            lazyllm.LOG.info(f"Loading StableDiffusion3Pipeline from {self.base_sd}")
            self.paintor = diffusers.StableDiffusion3Pipeline.from_pretrained(
                self.base_sd, torch_dtype=torch.float16)
            lazyllm.LOG.info(f"Pipeline loaded successfully, checking components...")
            # 检查关键组件是否已加载
            components = ['transformer', 'vae', 'text_encoder', 'tokenizer']
            for comp in components:
                if hasattr(self.paintor, comp):
                    comp_obj = getattr(self.paintor, comp)
                    if comp_obj is not None:
                        lazyllm.LOG.info(f"Component '{comp}' loaded: {type(comp_obj).__name__}")
                    else:
                        lazyllm.LOG.warning(f"Component '{comp}' is None!")
                else:
                    lazyllm.LOG.warning(f"Component '{comp}' not found in pipeline!")
            if self.num_gpus > 1:
                self.paintor.enable_model_cpu_offload()
                lazyllm.LOG.info(f"Loaded StableDiffusion3Pipeline with multi-GPU support (num_gpus={self.num_gpus})")
            else:
                self.paintor.to('cuda')
                lazyllm.LOG.info(f"Pipeline moved to cuda")
        except (ValueError, AttributeError) as e:
            # 如果 StableDiffusion3Pipeline 加载失败，尝试 ZImagePipeline
            lazyllm.LOG.warning(f"Failed to load StableDiffusion3Pipeline: {e}, trying ZImagePipeline")
            try:
                from diffusers import ZImagePipeline
                # 根据官方文档，使用 bfloat16 以获得最佳性能，并设置 low_cpu_mem_usage=False
                self.paintor = ZImagePipeline.from_pretrained(
                    self.base_sd,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=False,
                )
                if self.num_gpus > 1:
                    self.paintor.enable_model_cpu_offload()
                    lazyllm.LOG.info(f"Loaded ZImagePipeline with multi-GPU support (num_gpus={self.num_gpus})")
                else:
                    self.paintor.to('cuda')
            except Exception as e2:
                lazyllm.LOG.error(f"Failed to load ZImagePipeline: {e2}")
                raise

    @staticmethod
    def _validate_and_fix_images(images):
        """验证和修复图像，确保没有 NaN 或 inf 值"""
        if not images:
            lazyllm.LOG.error("No images generated!")
            return []
        fixed_images = []
        for idx, img in enumerate(images):
            if img is None:
                lazyllm.LOG.error(f"Image {idx} is None!")
                continue
            if isinstance(img, PIL.Image.Image):
                # 转换为 numpy 数组进行检查
                img_array = np.array(img)
                # 检查图像统计信息
                img_min, img_max = img_array.min(), img_array.max()
                img_mean = img_array.mean()
                img_std = img_array.std()
                lazyllm.LOG.info(f"Image {idx}: size={img.size}, mode={img.mode}, min={img_min}, max={img_max}, mean={img_mean:.2f}, std={img_std:.2f}")
                
                # 检查是否有 NaN 或 inf
                if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
                    lazyllm.LOG.warning(f"Image {idx} contains NaN or inf values, fixing...")
                    # 将 NaN 和 inf 替换为 0
                    img_array = np.nan_to_num(img_array, nan=0.0, posinf=255.0, neginf=0.0)
                    # 确保值在有效范围内
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                    img = PIL.Image.fromarray(img_array)
                # 如果图像值范围异常（全黑或全白），记录警告
                elif img_max == 0:
                    lazyllm.LOG.error(f"Image {idx} appears to be all black (max=0). This indicates a generation failure.")
                    lazyllm.LOG.error(f"Image stats: size={img.size}, mode={img.mode}, min={img_min}, max={img_max}, mean={img_mean:.2f}, std={img_std:.2f}")
                    # 不抛出异常，继续处理，让调用者决定如何处理
                elif img_min == 255:
                    lazyllm.LOG.warning(f"Image {idx} appears to be all white (min=255)")
                # 检查图像是否异常小（可能是生成失败）
                elif img_max - img_min < 10 and img_mean < 5:
                    lazyllm.LOG.warning(f"Image {idx} has very low variance (max-min={img_max-img_min}, mean={img_mean:.2f}), may be invalid")
                # 确保图像是 RGB 模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                fixed_images.append(img)
            else:
                lazyllm.LOG.warning(f"Image {idx} is not a PIL.Image.Image, type={type(img)}")
                fixed_images.append(img)
        return fixed_images

    @staticmethod
    def image_to_base64(image):
        if isinstance(image, PIL.Image.Image):
            # 确保图像是 RGB 模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            buffered = BytesIO()
            image.save(buffered, format='PNG')
            img_bytes = buffered.getvalue()
            img_size = len(img_bytes)
            lazyllm.LOG.info(f"Image saved to BytesIO: size={img_size} bytes, image_size={image.size}, mode={image.mode}")
            if img_size < 1000:  # 小于1KB的图像可能有问题
                lazyllm.LOG.warning(f"Image size is suspiciously small: {img_size} bytes")
            img_str = base64.b64encode(img_bytes).decode('utf-8')
            base64_len = len(img_str)
            lazyllm.LOG.info(f"Base64 encoded: length={base64_len} chars")
        elif isinstance(image, np.ndarray):
            # 确保数组值在有效范围内
            if np.any(np.isnan(image)) or np.any(np.isinf(image)):
                image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
            image = np.clip(image, 0, 255).astype(np.uint8)
            image = PIL.Image.fromarray(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            buffered = BytesIO()
            image.save(buffered, format='PNG')
            img_bytes = buffered.getvalue()
            img_size = len(img_bytes)
            lazyllm.LOG.info(f"Image (from numpy) saved to BytesIO: size={img_size} bytes, image_size={image.size}, mode={image.mode}")
            if img_size < 1000:
                lazyllm.LOG.warning(f"Image size is suspiciously small: {img_size} bytes")
            img_str = base64.b64encode(img_bytes).decode('utf-8')
            base64_len = len(img_str)
            lazyllm.LOG.info(f"Base64 encoded: length={base64_len} chars")
        else:
            raise ValueError('Unsupported image type')
        return f'data:image/png;base64,{img_str}'

    @staticmethod
    def images_to_base64(images):
        return [_StableDiffusion3.image_to_base64(img) for img in images]

    @staticmethod
    def image_to_file(image, file_path):
        if isinstance(image, PIL.Image.Image):
            image.save(file_path, format='PNG')
        elif isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)
            image.save(file_path, format='PNG')
        else:
            raise ValueError('Unsupported image type')

    @staticmethod
    def images_to_files(images, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        _delete_old_files(directory)
        unique_id = uuid.uuid4()
        path_list = []
        for i, img in enumerate(images):
            file_path = os.path.join(directory, f'image_{unique_id}_{i}.png')
            _StableDiffusion3.image_to_file(img, file_path)
            path_list.append(file_path)
        return path_list

    def __call__(self, string):
        lazyllm.call_once(self.init_flag, self.load_sd)
        # 处理 dict 类型的输入，提取实际的字符串
        if isinstance(string, dict):
            string = string.get('inputs') or string.get('prompt') or string.get('input') or \
                     next((v for v in string.values() if isinstance(v, str)), str(list(string.values())[0]) if string else '')

        for model_type, caller in self._call_registry.items():
            if model_type in self.base_sd.lower():
                lazyllm.LOG.info(f"Using registered caller for model type: {model_type}")
                return caller(self, string)

        # 检查 pipeline 类型，如果是 ZImagePipeline，使用不同的参数
        pipeline_type = type(self.paintor).__name__
        if 'ZImage' in pipeline_type:
            # ZImagePipeline 的调用方式（根据官方文档，Turbo 模型使用 guidance_scale=0.0 和更少的步数）
            result = self.paintor(
                prompt=string,
                height=1024,
                width=1024,
                num_inference_steps=9,  # Turbo 模型推荐值
                guidance_scale=0.0,  # Turbo 模型必须使用 0.0
                negative_prompt=None,
            )
            imgs = result.images
            # 验证和修复图像
            imgs = self._validate_and_fix_images(imgs)
        else:
            # StableDiffusion3Pipeline 的调用方式
            lazyllm.LOG.info(f"Calling StableDiffusion3Pipeline with prompt: {string[:100]}...")
            lazyllm.LOG.info(f"Pipeline type: {type(self.paintor).__name__}, base_sd: {self.base_sd}")
            # 检查 pipeline 是否已加载
            if not hasattr(self, 'paintor') or self.paintor is None:
                lazyllm.LOG.error("Pipeline not loaded!")
                raise ValueError("Pipeline not loaded")
            try:
                # 检查设备
                if hasattr(self.paintor, 'device'):
                    lazyllm.LOG.info(f"Pipeline device: {self.paintor.device}")
                # 生成图像
                result = self.paintor(
                    string,
                    negative_prompt='',
                    num_inference_steps=28,
                    guidance_scale=7.0,
                    max_sequence_length=512,
                )
                lazyllm.LOG.info(f"Generation result type: {type(result)}, has 'images' attr: {hasattr(result, 'images')}")
                if hasattr(result, 'images'):
                    imgs = result.images
                    lazyllm.LOG.info(f"Generated {len(imgs)} images, pipeline_type={type(self.paintor).__name__}")
                    if not imgs:
                        lazyllm.LOG.error("No images in result.images!")
                        raise ValueError("No images generated")
                    # 检查第一张图像的基本信息
                    if imgs and len(imgs) > 0:
                        first_img = imgs[0]
                        lazyllm.LOG.info(f"First image type: {type(first_img)}, size: {first_img.size if hasattr(first_img, 'size') else 'N/A'}, mode: {first_img.mode if hasattr(first_img, 'mode') else 'N/A'}")
                        # 快速检查是否为全黑
                        if isinstance(first_img, PIL.Image.Image):
                            img_array = np.array(first_img)
                            img_min, img_max = img_array.min(), img_array.max()
                            if img_max == 0:
                                lazyllm.LOG.error(f"Generated image is all black! This indicates generation failure.")
                                # 尝试检查是否有其他属性包含有用信息
                                if hasattr(result, 'nsfw_content_detected'):
                                    lazyllm.LOG.info(f"NSFW content detected: {result.nsfw_content_detected}")
                                # 检查模型组件状态
                                lazyllm.LOG.error("Checking model components for issues...")
                                if hasattr(self.paintor, 'transformer'):
                                    transformer = self.paintor.transformer
                                    if transformer is not None:
                                        lazyllm.LOG.info(f"Transformer: {type(transformer).__name__}, device: {next(transformer.parameters()).device if hasattr(transformer, 'parameters') else 'N/A'}")
                                    else:
                                        lazyllm.LOG.error("Transformer is None!")
                                if hasattr(self.paintor, 'vae'):
                                    vae = self.paintor.vae
                                    if vae is not None:
                                        lazyllm.LOG.info(f"VAE: {type(vae).__name__}, device: {next(vae.parameters()).device if hasattr(vae, 'parameters') else 'N/A'}")
                                    else:
                                        lazyllm.LOG.error("VAE is None!")
                                # 尝试使用备用参数重新生成
                                lazyllm.LOG.warning("Attempting regeneration with alternative parameters...")
                                try:
                                    result_retry = self.paintor(
                                        string,
                                        negative_prompt='',
                                        num_inference_steps=50,  # 增加步数
                                        guidance_scale=7.5,  # 稍微提高引导强度
                                        max_sequence_length=512,
                                    )
                                    if hasattr(result_retry, 'images') and result_retry.images:
                                        retry_img = result_retry.images[0]
                                        if isinstance(retry_img, PIL.Image.Image):
                                            retry_array = np.array(retry_img)
                                            retry_min, retry_max = retry_array.min(), retry_array.max()
                                            if retry_max > 0:
                                                lazyllm.LOG.info(f"Retry successful! New image range: min={retry_min}, max={retry_max}")
                                                imgs = result_retry.images
                                            else:
                                                lazyllm.LOG.error("Retry also produced all black image. This suggests a fundamental issue with the model or generation process.")
                                                # 检查是否是VAE解码问题
                                                lazyllm.LOG.error("Possible causes: 1) VAE decoder issue, 2) Model weights corrupted, 3) Device/memory issue, 4) Incompatible diffusers version")
                                except Exception as retry_e:
                                    lazyllm.LOG.error(f"Retry generation failed: {retry_e}", exc_info=True)
                else:
                    lazyllm.LOG.error(f"Result does not have 'images' attribute. Available attributes: {dir(result)}")
                    raise ValueError("Result does not have 'images' attribute")
            except Exception as e:
                lazyllm.LOG.error(f"Error during image generation: {e}", exc_info=True)
                raise
            # 验证和修复图像
            imgs = self._validate_and_fix_images(imgs)
            lazyllm.LOG.info(f"After validation: {len(imgs)} images")
            if not imgs:
                lazyllm.LOG.error("No images after validation!")
                raise ValueError("No valid images after validation")
            # 检查是否所有图像都是全黑的
            all_black = True
            for idx, img in enumerate(imgs):
                if isinstance(img, PIL.Image.Image):
                    img_array = np.array(img)
                    if img_array.max() > 0:
                        all_black = False
                        break
            if all_black:
                lazyllm.LOG.error("All generated images are black! This indicates a fundamental generation failure.")
                lazyllm.LOG.error("Possible causes:")
                lazyllm.LOG.error("1) VAE decoder issue - VAE may not be decoding latents correctly")
                lazyllm.LOG.error("2) Model weights corrupted or incomplete")
                lazyllm.LOG.error("3) Device/memory issue - GPU memory may be insufficient")
                lazyllm.LOG.error("4) Incompatible diffusers version")
                lazyllm.LOG.error("5) Model configuration issue")
                # 抛出异常，而不是返回全黑图像
                raise RuntimeError("Image generation failed: all generated images are black. Please check model files, device configuration, and diffusers version compatibility.")
        img_path_list = self.images_to_base64(imgs)
        lazyllm.LOG.info(f"Converted to base64: {len(img_path_list)} items, first item length={len(img_path_list[0]) if img_path_list else 0}")
        if img_path_list and len(img_path_list[0]) < 1000:
            lazyllm.LOG.warning(f"Base64 string is suspiciously short: {len(img_path_list[0])} chars")
        return encode_query_with_filepaths(files=img_path_list)

    @classmethod
    def rebuild(cls, base_sd, embed_batch_size, init, save_path, num_gpus=1):
        return cls(base_sd, embed_batch_size=embed_batch_size, init=init, save_path=save_path, num_gpus=num_gpus)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return _StableDiffusion3.rebuild, (self.base_sd, self.embed_batch_size, init, self.save_path, self.num_gpus)

@_StableDiffusion3.register_loader('flux')
def load_flux(model):
    import torch
    from diffusers import FluxPipeline
    # 多 GPU 支持：使用 enable_model_cpu_offload 自动分配
    # TODO: 目前只是显存分布式，不是计算分布式，待优化为真正的并行计算
    model.paintor = FluxPipeline.from_pretrained(
        model.base_sd, torch_dtype=torch.bfloat16)
    if model.num_gpus > 1:
        model.paintor.enable_model_cpu_offload()
        lazyllm.LOG.info(f"Loaded FluxPipeline with multi-GPU support (num_gpus={model.num_gpus})")
    else:
        model.paintor.to('cuda')

@_StableDiffusion3.register_caller('flux')
def call_flux(model, prompt):
    imgs = model.paintor(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.5,
        max_sequence_length=512,
    ).images
    # 验证和修复图像
    imgs = model._validate_and_fix_images(imgs)
    img_path_list = model.images_to_files(imgs, model.save_path)
    return encode_query_with_filepaths(files=img_path_list)

@_StableDiffusion3.register_loader('cogview')
def load_cogview(model):
    import torch
    from diffusers import CogView4Pipeline
    # 多 GPU 支持：使用 enable_model_cpu_offload 自动分配
    # TODO: 目前只是显存分布式，不是计算分布式，待优化为真正的并行计算
    model.paintor = CogView4Pipeline.from_pretrained(
        model.base_sd, torch_dtype=torch.bfloat16)
    if model.num_gpus > 1:
        model.paintor.enable_model_cpu_offload()
        lazyllm.LOG.info(f"Loaded CogView4Pipeline with multi-GPU support (num_gpus={model.num_gpus})")
    else:
        model.paintor.to('cuda')

@_StableDiffusion3.register_caller('cogview')
def call_cogview(model, prompt):
    imgs = model.paintor(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=3.5,
        num_images_per_prompt=1,
    ).images
    # 验证和修复图像
    imgs = model._validate_and_fix_images(imgs)
    img_path_list = model.images_to_files(imgs, model.save_path)
    return encode_query_with_filepaths(files=img_path_list)

@_StableDiffusion3.register_loader('zimage')
def load_zimage(model):
    import torch
    from diffusers import ZImagePipeline
    # 多 GPU 支持：使用 enable_model_cpu_offload 自动分配
    # TODO: 目前只是显存分布式，不是计算分布式，待优化为真正的并行计算
    # 根据官方文档，使用 bfloat16 以获得最佳性能，并设置 low_cpu_mem_usage=False
    model.paintor = ZImagePipeline.from_pretrained(
        model.base_sd,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    if model.num_gpus > 1:
        model.paintor.enable_model_cpu_offload()
        lazyllm.LOG.info(f"Loaded ZImagePipeline with multi-GPU support (num_gpus={model.num_gpus})")
    else:
        model.paintor.to('cuda')

@_StableDiffusion3.register_caller('zimage')
def call_zimage(model, prompt):
    # 根据官方文档，Turbo 模型使用 guidance_scale=0.0 和更少的步数
    result = model.paintor(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,  # Turbo 模型推荐值
        guidance_scale=0.0,  # Turbo 模型必须使用 0.0
        negative_prompt=None,
    )
    imgs = result.images
    # 验证和修复图像
    imgs = model._validate_and_fix_images(imgs)
    img_path_list = model.images_to_base64(imgs)
    return encode_query_with_filepaths(files=img_path_list)

@_StableDiffusion3.register_loader('hunyuanvideo')
def load_hunyuanvideo(model):
    import torch
    from diffusers import HunyuanVideo15Pipeline
    # HunyuanVideo 需要使用专门的 HunyuanVideo15Pipeline
    model.paintor = HunyuanVideo15Pipeline.from_pretrained(
        model.base_sd,
        torch_dtype=torch.bfloat16,
    )
    # 启用 VAE tiling 以节省显存
    if hasattr(model.paintor, 'vae') and hasattr(model.paintor.vae, 'enable_tiling'):
        model.paintor.vae.enable_tiling()
        lazyllm.LOG.info("Enabled VAE tiling for memory optimization")
    
    # 使用 sequential CPU offload 以更激进地节省显存
    # 这会逐个组件 offload，而不是整个模型
    if hasattr(model.paintor, 'enable_sequential_cpu_offload'):
        model.paintor.enable_sequential_cpu_offload()
        lazyllm.LOG.info("Enabled sequential CPU offload for maximum memory savings")
    elif model.num_gpus > 1:
        model.paintor.enable_model_cpu_offload()
        lazyllm.LOG.info(f"Loaded HunyuanVideo15Pipeline with multi-GPU support (num_gpus={model.num_gpus})")
    else:
        model.paintor.enable_model_cpu_offload()  # 即使单GPU也使用 offload 以节省显存
        lazyllm.LOG.info(f"Loaded HunyuanVideo15Pipeline on single GPU with CPU offload")

@_StableDiffusion3.register_caller('hunyuanvideo')
def call_hunyuanvideo(model, prompt):
    from diffusers.utils import export_to_video
    import torch
    # 确保 prompt 是字符串或列表类型
    if not isinstance(prompt, (str, list)):
        prompt = str(prompt)
    
    # 显存优化：降低帧数和步数
    # 121 帧需要 ~50GB 显存，49 帧需要 ~20GB 显存
    # 根据可用显存动态调整
    try:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            if gpu_memory < 50:
                # 40GB GPU: 使用更保守的参数
                num_frames = 49  # 约 2 秒 @ 24fps
                num_inference_steps = 30
                lazyllm.LOG.info(f"GPU memory: {gpu_memory:.1f}GB, using conservative settings: {num_frames} frames, {num_inference_steps} steps")
            elif gpu_memory < 80:
                # 50-80GB GPU: 中等参数
                num_frames = 81  # 约 3.4 秒 @ 24fps
                num_inference_steps = 40
                lazyllm.LOG.info(f"GPU memory: {gpu_memory:.1f}GB, using medium settings: {num_frames} frames, {num_inference_steps} steps")
            else:
                # 80GB+ GPU: 使用推荐参数
                num_frames = 121  # 约 5 秒 @ 24fps
                num_inference_steps = 50
                lazyllm.LOG.info(f"GPU memory: {gpu_memory:.1f}GB, using recommended settings: {num_frames} frames, {num_inference_steps} steps")
        else:
            num_frames = 49
            num_inference_steps = 30
    except Exception as e:
        lazyllm.LOG.warning(f"Failed to detect GPU memory, using conservative settings: {e}")
        num_frames = 49
        num_inference_steps = 30
    
    # HunyuanVideo15Pipeline 的正确调用方式
    # 根据官方文档，使用 num_frames 而不是 height/width
    generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 尝试使用 attention backend 优化（如果可用）
    # 注意：如果 flash-attn2 需要从 Hub 下载但失败，会回退到默认 attention
    try:
        from diffusers import attention_backend
        # 使用 flash attention 优化显存
        with attention_backend("flash_hub"):  # 或 "_flash_3_hub" for H100/H800
            result = model.paintor(
                prompt=prompt,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )
            lazyllm.LOG.info("Used flash attention backend for memory optimization")
    except Exception as e:
        # 捕获所有异常（包括 ImportError, AttributeError, 以及下载失败等）
        error_msg = str(e).lower()
        if 'snapshot folder' in error_msg or 'cannot find' in error_msg or 'hub' in error_msg:
            lazyllm.LOG.warning(f"Flash attention backend download failed or not available: {e}, using default attention")
        else:
            lazyllm.LOG.warning(f"Flash attention backend not available: {e}, using default attention")
        # 使用默认 attention
        result = model.paintor(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
    
    # HunyuanVideo15Pipeline 返回的 frames 是列表或数组
    # 注意：不能直接对数组进行布尔判断，需要使用 len() 或 is not None
    if hasattr(result, 'frames') and result.frames is not None:
        # 检查 frames 是否为空（支持列表、数组等可迭代对象）
        try:
            if len(result.frames) == 0:
                raise ValueError(f"HunyuanVideo15Pipeline returned empty frames")
            videos = result.frames
        except (TypeError, AttributeError):
            # 如果 result.frames 不支持 len()，尝试直接使用
            videos = result.frames
    else:
        raise ValueError(f"Unexpected result type from HunyuanVideo15Pipeline: {type(result)}, has 'frames': {hasattr(result, 'frames')}, frames value: {getattr(result, 'frames', None)}")
    
    unique_id = uuid.uuid4()
    if not os.path.exists(model.save_path):
        os.makedirs(model.save_path)
    vid_path_list = []
    for i, vid in enumerate(videos):
        file_path = os.path.join(model.save_path, f'video_{unique_id}_{i}.mp4')
        export_to_video(vid, file_path, fps=24)  # HunyuanVideo 推荐使用 24fps
        vid_path_list.append(file_path)
    return encode_query_with_filepaths(files=vid_path_list)

@_StableDiffusion3.register_loader('wan')
def load_wan(model):
    import torch
    from diffusers import WanPipeline, AutoencoderKLWan
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from transformers import T5Tokenizer, T5EncoderModel
    import os
    
    # tokenizer 和 text_encoder 在 google/umt5-xxl/ 子目录中
    tokenizer_path = os.path.join(model.base_sd, 'google', 'umt5-xxl')
    if not os.path.exists(tokenizer_path):
        lazyllm.LOG.warning(f"Tokenizer path {tokenizer_path} does not exist, using base_sd {model.base_sd}+'/tokenizer'")
        tokenizer_path = model.base_sd + '/tokenizer'
    # 确保 tokenizer_path 是字符串
    if not isinstance(tokenizer_path, str):
        tokenizer_path = str(tokenizer_path)
    
    # 先手动加载 tokenizer，确保路径正确
    # 这样可以避免 WanPipeline.from_pretrained 自动加载 tokenizer 时的路径问题
    tokenizer = None
    try:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        lazyllm.LOG.info(f"Loaded tokenizer from {tokenizer_path}")
    except Exception as e:
        lazyllm.LOG.warning(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        # 如果失败，尝试从根目录加载
        try:
            tokenizer = T5Tokenizer.from_pretrained(model.base_sd)
            lazyllm.LOG.info(f"Loaded tokenizer from {model.base_sd}")
        except Exception as e2:
            lazyllm.LOG.warning(f"Failed to load tokenizer from {model.base_sd}: {e2}")
            raise RuntimeError(f"Failed to load tokenizer: {e2}")
    
    # 先尝试自动加载 pipeline（不传递 tokenizer，让 pipeline 自动加载其他组件）
    try:
        # 多 GPU 支持：使用 enable_model_cpu_offload 自动分配
        # TODO: 目前只是显存分布式，不是计算分布式，待优化为真正的并行计算
        model.paintor = WanPipeline.from_pretrained(
            model.base_sd, 
            torch_dtype=torch.bfloat16)
        if model.num_gpus > 1:
            model.paintor.enable_model_cpu_offload()
            lazyllm.LOG.info(f"Loaded WanPipeline with multi-GPU support (num_gpus={model.num_gpus})")
        # 替换为我们手动加载的 tokenizer（确保路径正确）
        if hasattr(model.paintor, 'tokenizer') and tokenizer is not None:
            model.paintor.tokenizer = tokenizer
            lazyllm.LOG.info("Replaced pipeline tokenizer with manually loaded tokenizer")
            # 如果使用了 enable_model_cpu_offload，重新初始化以包含新的 tokenizer
            if model.num_gpus > 1:
                model.paintor.enable_model_cpu_offload()
    except Exception as e:
        # 如果自动加载失败，手动加载所有组件
        lazyllm.LOG.warning(f"WanPipeline auto-loading failed: {e}, loading components manually")
        
        # 手动加载各个组件
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        
        # text_encoder 的模型文件在根目录：models_t5_umt5-xxl-enc-bf16.pth
        # 优先使用 ModelScope，如果失败则尝试从本地权重文件直接加载
        text_encoder_model_file = os.path.join(model.base_sd, 'models_t5_umt5-xxl-enc-bf16.pth')
        
        text_encoder = None
        try:
            # 优先尝试从 ModelScope 加载 henjicc/umt5_xxl_encoder（更适合的模型路径）
            try:
                from modelscope import snapshot_download
                # 安全地获取 cache_dir，避免 KeyError
                try:
                    cache_dir = lazyllm.config['model_cache_dir']
                except (KeyError, TypeError):
                    cache_dir = None
                modelscope_model_path = snapshot_download('henjicc/umt5_xxl_encoder', cache_dir=cache_dir)
                text_encoder = T5EncoderModel.from_pretrained(
                    modelscope_model_path,
                    torch_dtype=torch.bfloat16,
                    local_files_only=False  # 允许在下载过程中继续等待
                )
                lazyllm.LOG.info(f"Loaded text_encoder from ModelScope: {modelscope_model_path}")
            except Exception as e1:
                lazyllm.LOG.warning(f"Failed to load from ModelScope henjicc/umt5_xxl_encoder: {e1}, trying AI-ModelScope/umt5-xxl")
                # 如果失败，尝试 AI-ModelScope/umt5-xxl
                try:
                    from modelscope import snapshot_download
                    try:
                        cache_dir = lazyllm.config['model_cache_dir']
                    except (KeyError, TypeError):
                        cache_dir = None
                    modelscope_model_path = snapshot_download('AI-ModelScope/umt5-xxl', cache_dir=cache_dir)
                    text_encoder = T5EncoderModel.from_pretrained(
                        modelscope_model_path,
                        torch_dtype=torch.bfloat16,
                        local_files_only=False  # 允许在下载过程中继续等待
                    )
                    lazyllm.LOG.info(f"Loaded text_encoder from ModelScope: {modelscope_model_path}")
                except Exception as e2:
                    lazyllm.LOG.warning(f"Failed to load from ModelScope AI-ModelScope/umt5-xxl: {e2}, trying Hugging Face")
                    # 如果 ModelScope 都失败，尝试 Hugging Face（但可能网络不可达）
                    try:
                        text_encoder = T5EncoderModel.from_pretrained(
                            'google/umt5-xxl',
                            torch_dtype=torch.bfloat16
                        )
                    except Exception as e3:
                        lazyllm.LOG.warning(f"Failed to load from Hugging Face: {e3}")
                        raise
            
            # 加载自定义权重文件（覆盖从 Hugging Face/ModelScope 加载的权重）
            if os.path.exists(text_encoder_model_file):
                lazyllm.LOG.info(f"Loading text_encoder weights from {text_encoder_model_file}")
                state_dict = torch.load(text_encoder_model_file, map_location='cpu')
                # 处理 state_dict 的键名（可能需要去除前缀）
                if isinstance(state_dict, dict):
                    # 检查是否有 'model' 或 'encoder' 前缀
                    if any(k.startswith('model.') for k in state_dict.keys()):
                        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                    elif any(k.startswith('encoder.') for k in state_dict.keys()):
                        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()}
                text_encoder.load_state_dict(state_dict, strict=False)
        except Exception as e:
            # 如果所有方法都失败，尝试直接从本地权重文件加载（需要手动创建配置）
            lazyllm.LOG.warning(f"Failed to load text_encoder from remote: {e}, trying direct loading from local file")
            if os.path.exists(text_encoder_model_file):
                try:
                    from transformers import T5Config
                    # 尝试从本地缓存或使用默认配置
                    # 先尝试从 google/umt5-xxl 目录加载配置（如果有）
                    config_path = os.path.join(model.base_sd, 'google', 'umt5-xxl', 'config.json')
                    if os.path.exists(config_path):
                        config = T5Config.from_pretrained(config_path)
                    else:
                        # 使用默认的 umt5-xxl 配置参数
                        config = T5Config(
                            vocab_size=250112,
                            d_model=4096,
                            d_kv=128,
                            d_ff=10240,
                            num_layers=24,
                            num_decoder_layers=24,
                            num_heads=64,
                            relative_attention_num_buckets=32,
                            relative_attention_max_distance=128,
                            dropout_rate=0.1,
                            layer_norm_epsilon=1e-6,
                            initializer_factor=1.0,
                            feed_forward_proj="gated-gelu",
                            is_encoder_decoder=True,
                            use_cache=True,
                            pad_token_id=0,
                            eos_token_id=1,
                            decoder_start_token_id=0,
                        )
                    text_encoder = T5EncoderModel(config)
                    state_dict = torch.load(text_encoder_model_file, map_location='cpu')
                    if isinstance(state_dict, dict):
                        if any(k.startswith('model.') for k in state_dict.keys()):
                            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                        elif any(k.startswith('encoder.') for k in state_dict.keys()):
                            state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()}
                    text_encoder.load_state_dict(state_dict, strict=False)
                    text_encoder = text_encoder.to(torch.bfloat16)
                    lazyllm.LOG.info("Successfully loaded text_encoder from local file with default config")
                except Exception as e2:
                    raise RuntimeError(f"Failed to load text_encoder from local file: {e2}")
            else:
                raise RuntimeError(f"text_encoder model file not found: {text_encoder_model_file}")
        
        # 加载 VAE（尝试从 vae 子目录或根目录）
        vae_path = os.path.join(model.base_sd, 'vae')
        if os.path.exists(vae_path):
            vae = AutoencoderKLWan.from_pretrained(
                model.base_sd, subfolder='vae', torch_dtype=torch.float32)
        else:
            # 尝试从根目录加载 VAE
            try:
                vae = AutoencoderKLWan.from_pretrained(
                    model.base_sd, torch_dtype=torch.float32)
            except Exception:
                vae = None
        
        # 加载 pipeline，先尝试完全自动加载（不传递任何组件）
        # 如果自动加载失败，再尝试传递已加载的组件
        try:
            # 多 GPU 支持：使用 enable_model_cpu_offload 自动分配
            # TODO: 目前只是显存分布式，不是计算分布式，待优化为真正的并行计算
            model.paintor = WanPipeline.from_pretrained(
                model.base_sd,
                torch_dtype=torch.bfloat16)
            if model.num_gpus > 1:
                model.paintor.enable_model_cpu_offload()
                lazyllm.LOG.info(f"Loaded WanPipeline with multi-GPU support (num_gpus={model.num_gpus})")
            # 如果自动加载成功，替换 tokenizer 和 text_encoder 为我们手动加载的版本
            if hasattr(model.paintor, 'tokenizer') and tokenizer is not None:
                model.paintor.tokenizer = tokenizer
            if hasattr(model.paintor, 'text_encoder') and text_encoder is not None:
                model.paintor.text_encoder = text_encoder
            if hasattr(model.paintor, 'vae') and vae is not None:
                model.paintor.vae = vae
            # 如果使用了 enable_model_cpu_offload，重新初始化以包含替换的组件
            if model.num_gpus > 1:
                model.paintor.enable_model_cpu_offload()
        except Exception as e:
            lazyllm.LOG.warning(f"WanPipeline auto-loading failed: {e}, trying with manual components")
            # 如果自动加载失败，尝试传递所有已加载的组件
            # 但需要确保所有必需组件都已加载
            raise RuntimeError(f"Failed to load WanPipeline: {e}. Please ensure all model files are downloaded.")
    
    # 设置自定义 scheduler
    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction',
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=3.0
    )
    model.paintor.scheduler = scheduler
    # 多 GPU 时，如果使用了 enable_model_cpu_offload，则不需要手动 to('cuda')
    # enable_model_cpu_offload 会自动管理设备分配
    if model.num_gpus <= 1:
        model.paintor.to('cuda')

@_StableDiffusion3.register_caller('wan')
def call_wan(model, prompt):
    from diffusers.utils import export_to_video
    # 确保 prompt 是字符串或列表类型（dict 已在 __call__ 中处理）
    if not isinstance(prompt, (str, list)):
        prompt = str(prompt)
    
    videos = model.paintor(
        prompt,
        negative_prompt=(
            'Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, '
            'static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, '
            'extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, '
            'fused fingers, still picture, messy background, three legs, '
            'many people in the background, walking backwards'),
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=5.0,
    ).frames
    unique_id = uuid.uuid4()
    if not os.path.exists(model.save_path):
        os.makedirs(model.save_path)
    vid_path_list = []
    for i, vid in enumerate(videos):
        file_path = os.path.join(model.save_path, f'video_{unique_id}_{i}.mp4')
        export_to_video(vid, file_path, fps=16)
        vid_path_list.append(file_path)
    return encode_query_with_filepaths(files=vid_path_list)


class StableDiffusionDeploy(LazyLLMDeployBase):
    message_format = None
    keys_name_handle = None
    default_headers = {'Content-Type': 'application/json'}

    @staticmethod
    def extract_result(x, inputs):
        # StableDiffusion 通过 RelayServer 返回，已经是 encode_query_with_filepaths 格式的字符串
        # 不需要像 Vllm 那样解析 JSON，直接返回即可
        return x

    def __init__(self, launcher: Optional[LazyLLMLaunchersBase] = None,
                 log_path: Optional[str] = None, trust_remote_code: bool = True, port: Optional[int] = None, **kw):
        super().__init__(launcher=launcher)
        self._log_path = log_path
        self._trust_remote_code = trust_remote_code
        self._port = port

    def __call__(self, finetuned_model=None, base_model=None):
        if not finetuned_model:
            finetuned_model = base_model
        elif not os.path.exists(finetuned_model) or \
            not any(file.endswith(('.bin', '.safetensors'))
                    for _, _, filenames in os.walk(finetuned_model) for file in filenames):
            LOG.warning(f'Note! That finetuned_model({finetuned_model}) is an invalid path, '
                        f'base_model({base_model}) will be used')
            finetuned_model = base_model
        # 从 launcher 获取 GPU 数量（如果可用）
        num_gpus = getattr(self._launcher, 'ngpus', 1) if self._launcher else 1
        num_gpus = max(1, int(num_gpus)) if num_gpus else 1
        return lazyllm.deploy.RelayServer(port=self._port, func=_StableDiffusion3(finetuned_model, num_gpus=num_gpus),
                                          launcher=self._launcher, log_path=self._log_path, cls='stable_diffusion')()
