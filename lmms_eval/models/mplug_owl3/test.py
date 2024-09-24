
import torch
# from lmms_eval.models.mplug_owl3.configuration_mplugowl3 import mPLUGOwl3Config
# from lmms_eval.models.mplug_owl3.modeling_mplugowl3 import mPLUGOwl3Model

from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoConfig
from decord import VideoReader, cpu    # pip install decord
model_path = '/data/mxy/HybridSSM/models/mPLUG-Owl3-7B-240728'
# config = mPLUGOwl3Config.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
print(config)
# model = mPLUGOwl3Model(config).cuda().half()
# model = mPLUGOwl3Model.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.half)
model = AutoModel.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.half,trust_remote_code=True)

model.eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)


messages = [
    {"role": "user", "content": """<|video|>
Describe this video."""},
    {"role": "assistant", "content": ""}
]

videos = ['/data/mxy/HybridSSM/Benchmark/videomme/data/_8lBR0E_Tx8.mp4']

MAX_NUM_FRAMES=16

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames
video_frames = [encode_video(_) for _ in videos]
inputs = processor(messages, images=None, videos=video_frames)

inputs.to('cuda')
inputs.update({
    'tokenizer': tokenizer,
    'max_new_tokens':100,
    'decode_text':True,
})


g = model.generate(**inputs)
print(g)
