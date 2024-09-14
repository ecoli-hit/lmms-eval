import os
import uuid
import base64
import warnings
from typing import List, Optional, Tuple, Union
import torch
import numpy as np
from io import BytesIO
from accelerate import Accelerator, DistributedType
from tqdm import tqdm
from PIL import Image
try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from mistral_common.protocol.instruct.messages import (
    UserMessage,
    TextChunk,
    ImageChunk,
    ImageURLChunk,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

@register_model("pixtral")
class Pixtral(lmms):
    """
    Pixtral Model based on pixtral-12b-240910
    """

    def __init__(
        self,
        pretrained: str = "pixtral",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        modality: str = "video",
        max_frames_num: int = 10,
        use_cache=True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Initialize Accelerator for multi-device
        accelerator = Accelerator()
        self._device = torch.device(f"cuda:{accelerator.local_process_index}") if accelerator.num_processes > 1 else device
        self._model = Transformer.from_folder("/data/mxy/models/Pixtral-12B-240910")
        self.modality = modality
        self.max_frames_num = max_frames_num

        # Initialize tokenizer for Pixtral
        self._tokenizer = MistralTokenizer.from_model(pretrained)
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
        assert False, "We have not implemented this function for pixtral yet"

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Generate text until a specific stopping condition or max token length is met.
        Each request has `args` with 6 elements: contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split.
        """
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # Process the visual input if present
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
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

            # Similar to llava, is visual paths has len 0
            # Then nothing will be executed

            request = ChatCompletionRequest(
                messages=[],
                model="pixtral",
            )

            # 创建消息时根据是否存在图片标记来处理上下文和图片
            if self.image_token not in contexts:
                # 没有图片标记时，将文本和图片一起添加到消息中
                message = UserMessage(
                    content=[
                        TextChunk(text=contexts)
                    ]
                )
                # 将所有图片添加到消息内容中
                for img in imgs:
                    message.content.append(
                        ImageURLChunk(image_url=f"data:image/png;base64,{img}")
                    )
                # 将消息添加到请求中
                request.messages.append(message)

            else:
                # 如果上下文中有图片标记，则分割文本并插入对应的图片
                contexts_split = contexts.split(self.image_token)
                for idx, img in enumerate(imgs):
                    message = UserMessage(
                        content=[
                            TextChunk(text=contexts_split[idx]),
                            ImageURLChunk(image_url=f"data:image/png;base64,{img}")
                        ]
                    )
                    # 将消息添加到请求中
                    request.messages.append(message)

            # Tokenize input
            tokenized = self.tokenizer.encode_chat_completion(request)
            tokens, text, images = tokenized.tokens, tokenized.text, tokenized.images
            
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
            
            

            out_tokens, _ = generate(
                [tokens], 
                self.model, 
                max_tokens=gen_kwargs["max_new_tokens"], 
                temperature=gen_kwargs["temperature"], 
                eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id)

            result = self.tokenizer.decode(out_tokens[0])

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
