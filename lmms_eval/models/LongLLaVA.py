import torch

torch.backends.cuda.matmul.allow_tf32 = True

import base64
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

from packaging import version
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria
from transformers import pipeline
from PIL import Image    
import requests
import numpy as np
warnings.filterwarnings("ignore")

from loguru import logger as eval_logger
from io import BytesIO
try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
except Exception as e:
    eval_logger.debug("LongLLaVA is not installed. Please install LongLLaVA to use this model.\nError: %s" % e)

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

# inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
# if is_flash_attn_2_available:
#     best_fit_attn_implementation = "flash_attention_2" # flash_attn has a bug that says: ERROR Error query and key must have the same dtype in generating

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"

from transformers import AutoModel, AutoTokenizer, AutoProcessor

from llava.model.builder import load_pretrained_model

from transformers import AutoProcessor, LlavaForConditionalGeneration


@register_model("LongLLaVA")
class LongLLaVA(lmms):
    """
    LongLLaVA Model
    """

    def __init__(
        self,
        pretrained: str = "/data/mxy/models/long-llava-qwen2-7b",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        modality: str = "video",
        max_frames_num: int = 10,
        use_cache=True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        model_id = "/data/mxy/models/long-llava-qwen2-7b"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
        ).to(0)

        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        # 将 tokenizer 和 model 应用到类中
        self._tokenizer = tokenizer
        self._model = model
        self._max_length = self._model.config.max_length
        accelerator = Accelerator()
        self._device = torch.device(f"cuda:{accelerator.local_process_index}") if accelerator.num_processes > 1 else torch.device(device)
        self.modality = modality
        self.max_frames_num = max_frames_num
        self.processor = processor
        # Initialize tokenizer for Pixtral
     #   self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.image_token = "<image>"
        # Set batch size per GPU and caching flag
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.accelerator = accelerator
        self._rank = accelerator.local_process_index if accelerator.num_processes > 1 else 0
        self._world_size = accelerator.num_processes if accelerator.num_processes > 1 else 1
        
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1
    
    # Function to encode the image
    def encode_image(self, image: Image):
        output_buffer = BytesIO()
        image.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "We have not implemented this function for LongLLaVA yet"


    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generate text until a specific stopping condition or max token length is met.
        Each request has `args` with 6 elements: contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split.
        """
        #requests = requests[:10]
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # Process the visual input if present
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            try:
                visuals = self.flatten(visuals)  # 确保 visuals 被展平
            except Exception as e:
                print(f"Error in generate_until: {e}")
                raise

            imgs = []  # multiple images or frames for video
            for visual in visuals:
                if self.modality == "image":
                    img = self.encode_image(visual)
                    imgs.append(img)
                elif self.modality == "video":
                    frames = self.encode_video(visual, self.max_frames_num)
                    imgs.extend(frames)


            if isinstance(contexts, tuple):
                contexts = list(contexts)

            # 构建消息内容并包含角色和内容
            message_content = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": contexts}
                    ]
                }
            ]

            for img in imgs:
                message_content[0]["content"].append(
                    {
                        "type": "image",
                        "image_url": f"data:image/png;base64,{img}"
                    }
                )


            # Tokenize input
            #tokenized = self.tokenizer.encode_chat_completion(request)
            #tokens, text, images = tokenized.tokens, tokenized.text, tokenized.images
            
            # 使用 batch_encode_plus 方法来编码 request 的内容
            tokenized = self.tokenizer.batch_encode_plus(
                [contexts],  # 使用 contexts 作为输入文本
                return_tensors='pt',  # 返回 PyTorch 张量
                padding=True,  # 自动填充
                truncation=True  # 自动截断
            )


            if "image_sizes" not in gen_kwargs:
                try:
                    gen_kwargs["image_sizes"] = [visuals[0].size]
                except:
                    gen_kwargs["image_sizes"] = None
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if "until" not in gen_kwargs:
                gen_kwargs["until"] = ["\n\n"]
            

            # 解码 base64 编码的图像为 PIL.Image 对象
            processed_imgs = []
            for img in imgs:
                img_data = base64.b64decode(img)  # 解码 base64
                img = Image.open(BytesIO(img_data))  # 转换为 PIL.Image 对象
                processed_imgs.append(img)
            
            prompt = self.processor.apply_chat_template(message_content, add_generation_prompt=True)
            inputs = self.processor(images=processed_imgs, text=prompt, return_tensors='pt').to(self._device, torch.float16)
            out_tokens = self._model.generate(**inputs, max_new_tokens=200, do_sample=False)

            #result = self.tokenizer.decode(out_tokens[0])
            #result = self.tokenizer.decode(out_tokens[0],skip_special_tokens=True)
            result = self.processor.decode(out_tokens[0][-2:-1], skip_special_tokens=True)
            print("result:",result)
            # Generate response based on the tokens and generation parameters
            res.append(result)

            pbar.update(1)

        pbar.close()
        return res
    
    def flatten(self, input):
        """Flatten nested list."""
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list