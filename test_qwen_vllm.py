import soundfile as sf
from io import BytesIO
from urllib.request import urlopen
from qwen_omni_utils import process_vision_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
import json
import os
from tqdm import tqdm
from jiwer import wer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import sys
from dataclasses import asdict
from typing import Any, NamedTuple, Optional

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.lora.request import LoRARequest
from vllm.utils import FlexibleArgumentParser

normalizer = BasicTextNormalizer()

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: Optional[str] = None
    prompt_token_ids: Optional[dict[str, list[int]]] = None
    multi_modal_data: Optional[dict[str, Any]] = None
    stop_token_ids: Optional[list[int]] = None
    lora_requests: Optional[list[LoRARequest]] = None

def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "audio language models"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1, help="Number of prompts to run."
    )
    parser.add_argument(
        "--num-audios",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Number of audio items per prompt.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Task name.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Checkpoint name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set the seed when initializing `vllm.LLM`.",
    )

    return parser.parse_args()


def run_qwen2_5_omni(question: str, audio_count: int, model_path: str):
    model_name = model_path

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"audio": audio_count},
    )

    audio_in_prompt = "".join(
        ["<|audio_bos|><|AUDIO|><|audio_eos|>\n" for idx in range(audio_count)]
    )

    default_system = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
        "Group, capable of perceiving auditory and visual inputs, as well as "
        "generating text and speech."
    )

    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{audio_in_prompt}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
    )

def main(args):


    model_path = f"./target_dir/{args.task}-whisper-5-best-with-audio-sharegpt-ckpt-{args.ckpt}"
    audio_count = args.num_audios


    if audio_count == 0:
        with open(f"data/sharegpt_data/{args.task}_test_whisper_5_best_no_audio.json", "r") as f:
            data = json.load(f)
    else:
        with open(f"data/sharegpt_data/{args.task}_test_whisper_5_best_with_audio.json", "r") as f:
            data = json.load(f)
    output_dir = f"output/{args.task}_test_with_audio_whisper_5_best_ckpt-{args.ckpt}" if audio_count > 0 else f"output/{args.task}_test_no_audio_whisper_5_best_ckpt-{args.ckpt}"
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(os.path.join(output_dir, "output.json"), "r") as f:
            exist_data = json.load(f)
    except:
        exist_data = []
    for i in tqdm(range(len(data))):
        # if i < len(exist_data):
        #     continue
        example = data[i]
        question = example['conversations'][0]['value']
        if audio_count > 0:
            audio_path = example['audios'][0]
        else:
            audio_path = None

        req_data = run_qwen2_5_omni(
            question, audio_count, model_path
        )
        # Disable other modalities to save memory
        default_limits = {"image": 0, "video": 0, "audio": 0}
        req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
            req_data.engine_args.limit_mm_per_prompt or {}
        )

        engine_args = asdict(req_data.engine_args) | {"seed": args.seed}
        llm = LLM(**engine_args)

        # We set temperature to 0.2 so that outputs can be different
        # even when all prompts are identical when running batch inference.
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=512, stop_token_ids=req_data.stop_token_ids
        )

        mm_data = req_data.multi_modal_data
  
        if not mm_data:
            mm_data = {}
            if audio_count > 0:
                print(audio_path)
                mm_data = {
                    "audio": [
                        audio_path
                    ]
                }
        # assert False
        assert args.num_prompts > 0
        inputs = {"multi_modal_data": mm_data}

        if req_data.prompt:
            inputs["prompt"] = req_data.prompt
        else:
            inputs["prompt_token_ids"] = req_data.prompt_token_ids

        if args.num_prompts > 1:
            # Batch inference
            inputs = [inputs] * args.num_prompts
        # Add LoRA request if applicable
        lora_request = (
            req_data.lora_requests * args.num_prompts if req_data.lora_requests else None
        )

        outputs = llm.generate(
            inputs,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

        for o in outputs:
            generated_text = o.outputs[0].text
            print(generated_text)
        if args.num_audios > 0:
            out_example = {
                "pred": normalizer(generated_text.split("assistant\n")[1]),
                "gold": normalizer(example['conversations'][1]['value']),
                "audio": example['audios'][0]
            }
        else:
            out_example = {
                "pred": normalizer(generated_text.split("assistant\n")[1]),
                "gold": normalizer(example['conversations'][1]['value']),
            }
        assert False
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

if __name__ == "__main__":
    args = parse_args()
    main(args)

