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

    def __init__(self, base_sd, source=None, embed_batch_size=30, trust_remote_code=True, save_path=None, init=False):
        source = lazyllm.config['model_source'] if not source else source
        self.base_sd = ModelManager(source).download(base_sd) or ''
        self.embed_batch_size = embed_batch_size
        self.trust_remote_code = trust_remote_code
        self.paintor = None
        self.init_flag = lazyllm.once_flag()
        self.save_path = save_path or os.path.join(lazyllm.config['temp_dir'], 'sd3')
        if init:
            lazyllm.call_once(self.init_flag, self.load_sd)

    def load_sd(self):
        if importlib.util.find_spec('torch_npu') is not None:
            import torch_npu  # noqa F401
            from torch_npu.contrib import transfer_to_npu  # noqa F401

        for model_type, loader in self._load_registry.items():
            if model_type in self.base_sd.lower():
                loader(self)
                return

        self.paintor = diffusers.StableDiffusion3Pipeline.from_pretrained(
            self.base_sd, torch_dtype=torch.float16).to('cuda')

    @staticmethod
    def image_to_base64(image):
        if isinstance(image, PIL.Image.Image):
            buffered = BytesIO()
            image.save(buffered, format='PNG')
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        elif isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)
            buffered = BytesIO()
            image.save(buffered, format='PNG')
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
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

        for model_type, caller in self._call_registry.items():
            if model_type in self.base_sd.lower():
                return caller(self, string)

        imgs = self.paintor(
            string,
            negative_prompt='',
            num_inference_steps=28,
            guidance_scale=7.0,
            max_sequence_length=512,
        ).images
        img_path_list = self.images_to_base64(imgs)
        return encode_query_with_filepaths(files=img_path_list)

    @classmethod
    def rebuild(cls, base_sd, embed_batch_size, init, save_path):
        return cls(base_sd, embed_batch_size=embed_batch_size, init=init, save_path=save_path)

    def __reduce__(self):
        init = bool(os.getenv('LAZYLLM_ON_CLOUDPICKLE', None) == 'ON' or self.init_flag)
        return _StableDiffusion3.rebuild, (self.base_sd, self.embed_batch_size, init, self.save_path)

@_StableDiffusion3.register_loader('flux')
def load_flux(model):
    import torch
    from diffusers import FluxPipeline
    model.paintor = FluxPipeline.from_pretrained(
        model.base_sd, torch_dtype=torch.bfloat16).to('cuda')

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
    img_path_list = model.images_to_files(imgs, model.save_path)
    return encode_query_with_filepaths(files=img_path_list)

@_StableDiffusion3.register_loader('cogview')
def load_cogview(model):
    import torch
    from diffusers import CogView4Pipeline
    model.paintor = CogView4Pipeline.from_pretrained(
        model.base_sd, torch_dtype=torch.bfloat16).to('cuda')

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
    img_path_list = model.images_to_files(imgs, model.save_path)
    return encode_query_with_filepaths(files=img_path_list)

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
        tokenizer_path = model.base_sd
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
        model.paintor = WanPipeline.from_pretrained(
            model.base_sd, 
            torch_dtype=torch.bfloat16)
        # 替换为我们手动加载的 tokenizer（确保路径正确）
        if hasattr(model.paintor, 'tokenizer') and tokenizer is not None:
            model.paintor.tokenizer = tokenizer
            lazyllm.LOG.info("Replaced pipeline tokenizer with manually loaded tokenizer")
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
            model.paintor = WanPipeline.from_pretrained(
                model.base_sd,
                torch_dtype=torch.bfloat16)
            # 如果自动加载成功，替换 tokenizer 和 text_encoder 为我们手动加载的版本
            if hasattr(model.paintor, 'tokenizer') and tokenizer is not None:
                model.paintor.tokenizer = tokenizer
            if hasattr(model.paintor, 'text_encoder') and text_encoder is not None:
                model.paintor.text_encoder = text_encoder
            if hasattr(model.paintor, 'vae') and vae is not None:
                model.paintor.vae = vae
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
    model.paintor.to('cuda')

@_StableDiffusion3.register_caller('wan')
def call_wan(model, prompt):
    from diffusers.utils import export_to_video
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
        return lazyllm.deploy.RelayServer(port=self._port, func=_StableDiffusion3(finetuned_model),
                                          launcher=self._launcher, log_path=self._log_path, cls='stable_diffusion')()
