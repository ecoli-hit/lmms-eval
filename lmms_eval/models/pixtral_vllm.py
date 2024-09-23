import os
import uuid
import base64
import warnings
from typing import List, Optional, Tuple, Union
from copy import deepcopy
import torch
import decord
import numpy as np
from io import BytesIO
from accelerate import Accelerator, DistributedType
from vllm import LLM, SamplingParams
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

@register_model("pixtral_vllm")
class Pixtral_vllm(lmms):
    """
    Pixtral Model based on pixtral-12b-240910
    """

    def __init__(
        self,
        pretrained: str = "pixtral_vllm",
        device: Optional[str] = "cuda",
        batch_size: Optional[Union[int, str]] = 1,
        modality: str = "video",
        max_frames_num: int = 64,
        use_cache=True,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Initialize Accelerator for multi-device
        accelerator = Accelerator()
        self._device = torch.device(f"cuda:{accelerator.local_process_index}") if accelerator.num_processes > 1 else device
        self._model = LLM(model="/data/mxy/models/Pixtral-12B-240910", tokenizer_mode="mistral", limit_mm_per_prompt={"image":max_frames_num},
          max_model_len=128000, enable_chunked_prefill=False, device=self._device)
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
        target_width=360
        target_height=240
        
        imgs=[]
        
        try:
            if type(video_path) == str:
                vr = VideoReader(video_path, ctx=cpu(0), width=target_width, height=target_height)
            else:
                vr = VideoReader(video_path[0], ctx=cpu(0), width=target_width, height=target_height)

            total_frame_num = len(vr)
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

            # Ensure the last frame is included
            if total_frame_num - 1 not in uniform_sampled_frames:
                uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

            frame_idx = uniform_sampled_frames.tolist()
            frames = vr.get_batch(frame_idx).asnumpy()
            for frame in frames:
                imgs.append(Image.fromarray(frame))
        except Exception as e:
            imgs=[Image.new("RGB", (target_width, target_height), (0, 0, 0))] * for_get_frames_num

        base64_frames = []
        for img in imgs:
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
            # flag = True
            for visual in visuals:
                if self.modality == "image":
                    img = self.encode_image(visual)
                    imgs.append(img)
                elif self.modality == "video":
                    frames = self.encode_video(visual, self.max_frames_num)
            #         if frames == False:
            #             flag = False
            #             break
                    imgs.extend(frames)
                    
            # if flag == False:
            #     res.append("No answer")
            #     pbar.update(1)
            #     continue

            messages=[]

            # If there is no image token in the context, append the image to the text
            if self.image_token not in contexts:
                # Create a message with only one image at a time
                for idx, img in enumerate(imgs):
                    message_content = [{"type": "text", "text": contexts}]
                    message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
                    
                    # Add each message separately to avoid multiple images in one request
                    messages.append({"role": "user", "content": message_content})
            else:
                # Split contexts by the image token
                contexts_split = contexts.split(self.image_token)
                
                # For each part of the context, pair it with the corresponding image
                for idx, img in enumerate(imgs):
                    message_content = [{"type": "text", "text": contexts_split[idx]}]
                    message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
                    
                    # Add each message separately to avoid multiple images in one request
                    messages.append({"role": "user", "content": message_content})

                # Add the last chunk of context without an image if it exists
                if len(contexts_split) > len(imgs):
                    messages.append({"role": "user", "content": [{"type": "text", "text": contexts_split[-1]}]})
            
            
            if "image_sizes" not in gen_kwargs:
                try:
                    gen_kwargs["image_sizes"] = [visuals[0].size]
                except:
                    gen_kwargs["image_sizes"] = None
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 8192
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            
            sampling_params = SamplingParams(
                max_tokens=gen_kwargs["max_new_tokens"],
                temperature=gen_kwargs["temperature"],
                use_beam_search=False,
                top_p=gen_kwargs["top_p"],
            )
            
            outputs = self._model.chat(messages=messages, sampling_params=sampling_params)
            
            result=outputs[0].outputs[0].text
            # print("result:",result)
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
