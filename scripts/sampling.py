import json


datasets = ["tedlium", "gigaspeech", "voxpopuli", "ami", "spgispeech"]

type = ["train", "test"]

for dataset in datasets:
    for t in type:
        with open(f"data/sharegpt_data/{dataset}_{t}_whisper_5_best_with_audio.json", "r") as f:
            data = json.load(f)
        print(f"Dataset: {dataset}, Type: {t}")
        print(len(data))
        if len(data) > 20000:
            data = data[:20000]
        with open(f"data/sharegpt_data/{dataset}_{t}_whisper_5_best_with_audio.json", "w") as f:
            json.dump(data, f, indent=1)