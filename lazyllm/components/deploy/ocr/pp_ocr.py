import os
import json
import lazyllm
from typing import Optional
import string
from ..base import LazyLLMDeployBase
from ...utils.file_operate import _base64_to_file, _is_base64_with_mime

punctuation = set(string.punctuation + '，。！？；：“”‘’（）【】《》…—～、')


def is_all_punctuation(s: str) -> bool:
    return all(c in punctuation for c in s)


class _OCR(object):
    def __init__(
        self,
        model: Optional[str] = 'PP-OCRv5_mobile',
        use_doc_orientation_classify: Optional[bool] = False,
        use_doc_unwarping: Optional[bool] = False,
        use_textline_orientation: Optional[bool] = False,
        device: Optional[str] = None,
        **kw
    ):
        self.model = model
        # 如果 model 已经包含 _det 或 _rec 后缀，不要重复添加
        # 支持传入完整模型名称如 'PP-OCRv5_server_det'
        if model.endswith('_det'):
            # 如果已经是 det 模型，提取基础名称
            base_model = model[:-4]  # 移除 '_det'
            self.text_detection_model_name = model  # 保持原样
            self.text_recognition_model_name = base_model + '_rec'
        elif model.endswith('_rec'):
            # 如果已经是 rec 模型，提取基础名称
            base_model = model[:-4]  # 移除 '_rec'
            self.text_detection_model_name = base_model + '_det'
            self.text_recognition_model_name = model  # 保持原样
        else:
            # 标准情况：添加后缀
            self.text_detection_model_name = model + '_det'
            self.text_recognition_model_name = model + '_rec'
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_textline_orientation = use_textline_orientation
        
        # 设备配置：如果没有指定，尝试自动检测
        if device is None:
            try:
                import subprocess
                import os
                result = subprocess.run(['nvidia-smi', '-L'], 
                                      capture_output=True, 
                                      timeout=2)
                if result.returncode == 0 and result.stdout:
                    device = 'gpu:0'
                elif os.environ.get('CUDA_VISIBLE_DEVICES'):
                    device = 'gpu:0'
                else:
                    try:
                        from lazyllm.thirdparty import paddle
                        if hasattr(paddle, 'device') and paddle.device.is_compiled_with_cuda():
                            device = 'gpu:0'
                        else:
                            device = 'cpu'
                    except ImportError:
                        device = 'cpu'
            except:
                device = 'cpu'
        
        self.device = device
        
        self.init_flag = lazyllm.once_flag()

    def load_paddleocr(self):
        import sys
        try:
            from lazyllm.thirdparty import paddleocr
            # 打印paddleocr版本（仅在成功导入时）
            if hasattr(paddleocr, '__version__'):
                lazyllm.LOG.info(f'PaddleOCR version: {paddleocr.__version__}')
        except ImportError as e:
            # 使用 lazyllm.LOG 记录错误，避免重复打印
            error_msg = (
                f"\n{'='*60}\n"
                f"❌ 无法导入 paddleocr 模块\n"
                f"{'='*60}\n"
                f"当前 Python 解释器: {sys.executable}\n"
                f"Python 版本: {sys.version}\n"
                f"Python 路径: {sys.path[:3]}...\n"
                f"\n请在该 Python 环境中安装 paddleocr:\n"
                f"  {sys.executable} -m pip install paddleocr\n"
                f"\n或者如果使用虚拟环境，请先激活虚拟环境再安装。\n"
                f"{'='*60}\n"
            )
            lazyllm.LOG.error(error_msg)
            raise ImportError(error_msg) from e
        # 初始化参数
        paddleocr_kwargs = {
            'use_doc_orientation_classify': self.use_doc_orientation_classify,
            'use_doc_unwarping': self.use_doc_unwarping,
            'use_textline_orientation': self.use_textline_orientation,
            'text_detection_model_name': self.text_detection_model_name,
            'device': self.device,
        }
        
        # 查找本地模型文件
        # 支持多种路径格式：
        # 1. model_path/PP-OCRv5_server_det/
        # 2. model_path/PaddlePaddle/PP-OCRv5_server_det/
        # 3. 直接传入的完整路径
        try:
            model_path = lazyllm.config['model_path']
        except (KeyError, RuntimeError):
            model_path = ''
        model_paths = model_path.split(':') if model_path else ['']
        
        # 尝试查找检测模型
        det_model_dir = None
        for base_path in model_paths:
            if not base_path:
                continue
            # 尝试多种路径格式
            possible_paths = [
                os.path.join(base_path, self.text_detection_model_name),
                os.path.join(base_path, 'PaddlePaddle', self.text_detection_model_name),
                os.path.join(base_path, 'modelscope', 'PaddlePaddle', self.text_detection_model_name),
            ]
            # 如果传入的 model 本身是完整路径
            if os.path.isabs(self.model) and os.path.exists(self.model):
                if self.model.endswith('_det'):
                    det_model_dir = self.model
                    break
            # 检查可能的路径，并验证是否为有效的模型目录
            for path in possible_paths:
                if os.path.exists(path) and self._is_valid_model_dir(path):
                    det_model_dir = path
                    break
            if det_model_dir:
                break
        
        # 如果找到本地模型目录，从配置文件中读取实际的模型名称
        if det_model_dir:
            actual_det_model_name = self._get_model_name_from_dir(det_model_dir)
            if actual_det_model_name:
                paddleocr_kwargs['text_detection_model_name'] = actual_det_model_name
            paddleocr_kwargs['det_model_dir'] = det_model_dir
        
        # 尝试查找识别模型
        rec_model_dir = None
        for base_path in model_paths:
            if not base_path:
                continue
            # 尝试多种路径格式
            possible_paths = [
                os.path.join(base_path, self.text_recognition_model_name),
                os.path.join(base_path, 'PaddlePaddle', self.text_recognition_model_name),
                os.path.join(base_path, 'modelscope', 'PaddlePaddle', self.text_recognition_model_name),
            ]
            # 检查可能的路径，并验证是否为有效的模型目录
            for path in possible_paths:
                if os.path.exists(path) and self._is_valid_model_dir(path):
                    rec_model_dir = path
                    break
            if rec_model_dir:
                break
        
        if rec_model_dir:
            actual_rec_model_name = self._get_model_name_from_dir(rec_model_dir)
            if actual_rec_model_name:
                paddleocr_kwargs['text_recognition_model_name'] = actual_rec_model_name
            else:
                paddleocr_kwargs['text_recognition_model_name'] = self.text_recognition_model_name
            paddleocr_kwargs['rec_model_dir'] = rec_model_dir

        self.ocr = paddleocr.PaddleOCR(**paddleocr_kwargs)
    
    def _is_valid_model_dir(self, model_dir):
        """验证模型目录是否有效（包含必要的模型文件）"""
        if not os.path.isdir(model_dir):
            return False
        
        # 检查是否包含模型文件（.pdiparams, .pdmodel, inference.pdiparams 等）
        required_files = [
            'inference.pdiparams',
            'inference.pdmodel',
        ]
        
        # 至少需要有一个模型文件
        has_model_file = False
        for file in required_files:
            if os.path.exists(os.path.join(model_dir, file)):
                has_model_file = True
                break
        
        # 或者检查是否有配置文件
        has_config = os.path.exists(os.path.join(model_dir, 'inference.yml')) or \
                     os.path.exists(os.path.join(model_dir, 'config.json'))
        
        return has_model_file or has_config
    
    def _get_model_name_from_dir(self, model_dir):
        """从模型目录的配置文件中读取模型名称"""
        try:
            import yaml
        except ImportError:
            yaml = None
        
        # 尝试读取 inference.yml
        if yaml:
            inference_yml = os.path.join(model_dir, 'inference.yml')
            if os.path.exists(inference_yml):
                try:
                    with open(inference_yml, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        if config and 'Global' in config and 'model_name' in config['Global']:
                            return config['Global']['model_name']
                except Exception as e:
                    lazyllm.LOG.warning(f'Failed to read inference.yml: {e}')
        
        # 尝试读取 config.json
        config_json = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_json):
            try:
                with open(config_json, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if config and 'Global' in config and 'model_name' in config['Global']:
                        return config['Global']['model_name']
            except Exception as e:
                lazyllm.LOG.warning(f'Failed to read config.json: {e}')
        
        return None

    def __call__(self, input):
        lazyllm.call_once(self.init_flag, self.load_paddleocr)
        if isinstance(input, dict):
            if 'inputs' in input:
                file_list = input['inputs']
            else:
                file_list = input
        else:
            try:
                file_list = lazyllm.components.formatter.formatterbase._lazyllm_get_file_list(input)
            except (TypeError, AttributeError) as e:
                # 如果 _lazyllm_get_file_list 失败，直接使用原始输入
                lazyllm.LOG.warning(f'_lazyllm_get_file_list failed: {e}, using raw input')
                file_list = input
        
        # 处理输入：可能是字符串、列表或字典
        if isinstance(file_list, str):
            # 尝试解析 JSON 字符串
            try:
                parsed = json.loads(file_list)
                if isinstance(parsed, dict):
                    if 'files' in parsed:
                        file_list = parsed['files']
                    elif 'inputs' in parsed:
                        file_list = parsed['inputs']
                    else:
                        # 尝试从字典值中提取文件路径
                        for value in parsed.values():
                            if isinstance(value, list) and len(value) > 0:
                                if all(isinstance(item, str) and (item.endswith(('.pdf', '.png', '.jpg', '.jpeg', '.bmp')) or os.path.sep in item) for item in value):
                                    file_list = value
                                    break
                        if isinstance(file_list, str):
                            file_list = [v for v in parsed.values() if isinstance(v, (str, list))]
                            if file_list and isinstance(file_list[0], list):
                                file_list = file_list[0]
                elif isinstance(parsed, list):
                    file_list = parsed
                else:
                    file_list = [file_list]
            except (json.JSONDecodeError, ValueError):
                file_list = [file_list]
        elif isinstance(file_list, dict):
            # 如果直接是字典，提取文件列表
            if 'files' in file_list:
                file_list = file_list['files']
            elif 'inputs' in file_list:
                file_list = file_list['inputs']
            else:
                # 尝试从字典值中提取文件路径
                for key, value in file_list.items():
                    if isinstance(value, list) and len(value) > 0:
                        if all(isinstance(item, str) and (item.endswith(('.pdf', '.png', '.jpg', '.jpeg', '.bmp')) or os.path.sep in item) for item in value):
                            file_list = value
                            break
        elif not isinstance(file_list, list):
            file_list = [file_list]
        
        # 确保 file_list 是列表
        if not isinstance(file_list, list):
            file_list = [file_list]
        
        # 展平嵌套列表
        flattened_files = []
        for item in file_list:
            if isinstance(item, list):
                flattened_files.extend(item)
            else:
                flattened_files.append(item)
        file_list = flattened_files
        
        # 处理文件列表：如果是 base64 格式则转换，否则当作文件路径
        processed_files = []
        for file in file_list:
            if not isinstance(file, str):
                continue
            
            # 清理路径（移除可能的引号或空格）
            file = file.strip().strip('"').strip("'")
            
            # 检查是否是 JSON 字符串
            if file.startswith('{') and '}' in file:
                try:
                    parsed = json.loads(file)
                    if isinstance(parsed, dict) and 'files' in parsed:
                        for nested_file in parsed['files']:
                            if isinstance(nested_file, str):
                                nested_file = nested_file.strip().strip('"').strip("'")
                                if _is_base64_with_mime(nested_file):
                                    processed_files.append(_base64_to_file(nested_file))
                                elif nested_file.endswith(('.pdf', '.png', '.jpg', '.jpeg', '.bmp')):
                                    processed_files.append(nested_file)
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
            
            if _is_base64_with_mime(file):
                processed_files.append(_base64_to_file(file))
            elif file.endswith(('.pdf', '.png', '.jpg', '.jpeg', '.bmp')):
                processed_files.append(file)
        
        if not processed_files:
            raise ValueError(f'No valid files found in input')
        
        file_list = processed_files
        txt = []
        for file in file_list:
            if hasattr(self.ocr, 'predict'):
                result = self.ocr.predict(file)
            else:
                result = self.ocr.ocr(file)
            for res in result:
                for sentence in res['rec_texts']:
                    t = sentence.strip()
                    if not is_all_punctuation(t) and len(t) > 0:
                        txt.append(t)
        return '\n'.join(txt)

    @classmethod
    def rebuild(cls, *args, **kw):
        return cls(*args, **kw)

    def __reduce__(self):
        return _OCR.rebuild, (
            self.model,
            self.use_doc_orientation_classify,
            self.use_doc_unwarping,
            self.use_textline_orientation,
        )


class OCRDeploy(LazyLLMDeployBase):
    keys_name_handle = {
        'inputs': 'inputs',
        'ocr_files': 'inputs',
    }
    message_format = {'inputs': '/path/to/pdf'}
    default_headers = {'Content-Type': 'application/json'}

    def __init__(self, launcher=None, log_path=None, trust_remote_code=True, port=None, **kw):
        super().__init__(launcher=launcher)
        self._log_path = log_path
        self._trust_remote_code = trust_remote_code
        self._port = port

    def __call__(self, finetuned_model=None, base_model=None):
        if not finetuned_model:
            finetuned_model = base_model
        return lazyllm.deploy.RelayServer(
            port=self._port, func=_OCR(finetuned_model), launcher=self._launcher, log_path=self._log_path, cls='ocr')()
