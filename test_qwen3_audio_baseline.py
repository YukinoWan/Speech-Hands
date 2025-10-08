import soundfile as sf
from io import BytesIO
from urllib.request import urlopen
import librosa
import numpy as np
import json
import os
from tqdm import tqdm
from jiwer import wer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import sys
import os
import torch


from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeForConditionalGeneration
from qwen_omni_utils import process_mm_info



def get_response_vllm(sound, processor, llm, sampling_params):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": sound},
                {"type": "text", "text": "Transcribe the English audio into text."}
            ], 
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        "mm_processor_kwargs": {
            "use_audio_in_video": True,
        },
    }

    if images is not None:
        inputs['multi_modal_data']['image'] = images
    if videos is not None:
        inputs['multi_modal_data']['video'] = videos
    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios

    outputs = llm.generate([inputs], sampling_params=sampling_params)

    print(outputs[0].outputs[0].text)
    return outputs[0].outputs[0].text

def get_response(sound, processor, model):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": sound},
                {"type": "text", "text": "Transcribe the English audio into text."}
            ],
        },
    ]

    # Set whether to use audio in video
    USE_AUDIO_IN_VIDEO = True

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, 
                    audio=audios, 
                    images=images, 
                    videos=videos, 
                    return_tensors="pt", 
                    padding=True, 
                    use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    text_ids, audio = model.generate(**inputs, 
                                 speaker="Ethan", 
                                 thinker_return_dict_in_generate=True,
                                 use_audio_in_video=USE_AUDIO_IN_VIDEO)

    text = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
    print(text[0])
    return text[0]
        

normalizer = BasicTextNormalizer()

if __name__ == '__main__':
    # vLLM engine v1 not supported yet
    os.environ['VLLM_USE_V1'] = '0'
    task = sys.argv[1]
    use_vllm = sys.argv[2]
    print(use_vllm)

    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    # MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    if use_vllm == "True":
        print("good")
        llm = LLM(
                model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.95,
                tensor_parallel_size=torch.cuda.device_count(),
                limit_mm_per_prompt={'image': 1, 'video': 3, 'audio': 3},
                max_num_seqs=1,
                max_model_len=32768,
                seed=1234,
        )

        sampling_params = SamplingParams(temperature=1e-2, top_p=0.1, top_k=1, max_tokens=8192)
    else:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        model.disable_talker()

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    #model_path = f"./target_dir/{task}-whisper-5-best-with-audio-sharegpt-ckpt-{ckpt}"

    # print(no_audio)

    with open(f"data/sharegpt_data_v6/{task}_whisper_5_best_with_audio.json", "r") as f:
        data = json.load(f)
    # data = data[:100]

    if use_vllm == "True":
        output_dir = f"vllm_baseline_output/{task}_vllm_baseline_with_audio_qwen3"
    else:
        output_dir = f"baseline_output/{task}_direct_baseline_with_audio_qwen3"
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(os.path.join(output_dir, "output.json"), "r") as f:
            exist_data = json.load(f)
    except:
        exist_data = []


    for i in tqdm(range(len(data))):
        if i < len(exist_data):
            continue
        example = data[i]
        
        if use_vllm == "True":
            text = get_response_vllm(example['audios'][0], processor, llm, sampling_params)
        else:
            text = get_response(example['audios'][0], processor, model)
        
        out_example = {
            "pred": normalizer(text),
            "gold": normalizer(example['conversations'][1]['value']),
            "audio": example['audios'][0]
        }

        exist_data.append(out_example)
        with open(os.path.join(output_dir, "output.json"), "w") as f:
            json.dump(exist_data, f, indent=1, ensure_ascii=False)

    # compute normalized WER, normalize to ignore punctuations
    pred_list = [item['pred'] for item in exist_data if item['gold'] != "ignore time segment in scoring"]
    gold_list = [item['gold'] for item in exist_data if item['gold'] != "ignore time segment in scoring"]
    wer_score = wer(gold_list, pred_list)
    print(f"WER: {wer_score}")
    wer_dict = {"wer": wer_score}

    with open(os.path.join(output_dir, "metric.json"), "w") as f:
        json.dump(wer_dict, f, indent=1, ensure_ascii=False)
