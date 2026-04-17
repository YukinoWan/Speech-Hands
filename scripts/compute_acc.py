import json
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
normalizer = BasicTextNormalizer()
from tqdm import tqdm
import sys
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

SPEECH_HANDS_OUTPUT = os.environ.get("SPEECH_HANDS_OUTPUT", "./output")

def plot_confusion_matrix(y_true, y_pred, labels, title, outpath,
                          normalize=True, cmap=None, fontsize=10):
    """
    y_true, y_pred: 列表或 1D ndarray（标签值）
    labels: 按显示顺序的类别列表（与你的 token_map.values() 一致）
    title: 图标题
    outpath: 输出文件路径（根据后缀保存 .png / .pdf 等）
    normalize: True 表示按行归一化（每一行和为 1）
    cmap: 颜色图（可留 None 使用默认）
    fontsize: 字体大小
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize and cm.sum(axis=1, keepdims=True).any():
        with np.errstate(all='ignore'):
            cm = cm / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    n_class = len(labels)
    # 根据类别数自适应图大小
    fig_w = max(4.0, min(0.45 * n_class, 10.0))
    fig_h = fig_w  # 正方形更适合矩阵
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize-1)

    # 坐标轴与刻度
    ax.set_title(title, fontsize=fontsize+2, pad=8)
    ax.set_xlabel("Predicted", fontsize=fontsize)
    ax.set_ylabel("Gold", fontsize=fontsize)
    ax.set_xticks(np.arange(n_class))
    ax.set_yticks(np.arange(n_class))
    ax.set_xticklabels(labels, fontsize=fontsize-1, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=fontsize-1)

    # 在每个格子写数值（归一化显示到小数或不归一化显示整数）
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(n_class):
        for j in range(n_class):
            val = cm[i, j]
            ax.text(j, i, format(val, fmt),
                    ha="center", va="center",
                    fontsize=fontsize-1,
                    color="white" if val > thresh else "black")

    ax.set_ylim(n_class-0.5, -0.5)  # 避免顶部被裁剪
    fig.tight_layout()

    # 确保目录存在并保存
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


ckpt = sys.argv[1]

with open(f"{SPEECH_HANDS_OUTPUT}/2025/with_audio_{ckpt}/output.json", "r") as f:
    data = json.load(f)

with open(f"{SPEECH_HANDS_OUTPUT}/2025/with_audio_ckpt-baseline_dev/output.json", "r") as f:
    internal_data = json.load(f)
with open(f"{SPEECH_HANDS_OUTPUT}/2025/with_audio_ckpt-flamingo_dev/output.json", "r") as f:
    external_data = json.load(f)


part1_acc = 0
part2_acc = 0
part3_acc = 0
part1_data = []
part2_data = []
part3_data = []
use_token = True

def acc(pred, gold):
    if use_token:
        if "<internal>" in pred and "<internal>" in gold:
            return 1
        if "<external>" in pred and "<external>" in gold:
            return 1
        if "<no_ref>" in pred and "<no_ref>" in gold:
            return 1

        if ">" in gold:
            gold = gold.split(">")[1].strip()
        if ">" in pred:
            pred = pred.split(">")[1].strip()

    # print(pred, gold)
    if type(pred) == list:
        # assert False
        for pred in pred:
            if normalizer(pred.strip()) == normalizer(gold):
                return 1
        return 0
    else:
        return 1 if normalizer(pred.strip()) == normalizer(gold.strip()) else 0

part1_token_pred_list = []
part2_token_pred_list = []
part3_token_pred_list = []
part1_token_gold_list = []
part2_token_gold_list = []
part3_token_gold_list = []

if use_token:
    # token_map = {"<internal>": 1, "<external>": 2, "<rewrite_guided>": 3, "<rewrite_free>": 4}
    token_map = {"<internal>": "<internal>", "<external>": "<external>", "<rewrite_guided>": "<rewrite_guided>", "<rewrite_free>": "<rewrite_free>"}
    token_map = {"<internal>": "<internal>", "<external>": "<external>", "<rewrite>": "<rewrite>"}

    # token_map = {"<no_ref>": 1, "<need_ref>": 2}
    digits = 3
num_internal = 0
num_external = 0
num_rewrite = 0
for i, item in tqdm(enumerate(data)):
    internal_pred = internal_data[i]['pred']
    external_pred = external_data[i]['pred']

    pred_tokens = [token_map[token] for token in token_map.keys() if token in item['pred']][0]
    gold_tokens = [token_map[token] for token in token_map.keys() if token in item['gold']][0]
    if "dev/audio" in item['audio']:
        part3_data.append(item)
        part3_acc += acc(item['pred'], item['gold'])
        # if acc(item['pred'], item['gold']) == 0:
        #     print(item['pred'], item['gold'])
        if use_token:
            part3_token_pred_list.append(pred_tokens)
            part3_token_gold_list.append(gold_tokens)
    elif "dev/fold" in item['audio']:
        part2_data.append(item)
        part2_acc += acc(item['pred'], item['gold'])
        # if acc(item['pred'], item['gold']) == 1:
        #     if normalizer(item['pred'][0].strip()) != normalizer(item['gold'].strip()):
        #         print(item['pred'], item['gold'])
        if use_token:
            part2_token_pred_list.append(pred_tokens)
            part2_token_gold_list.append(gold_tokens)
    else:
        part1_data.append(item)
        part1_acc += acc(item['pred'], item['gold'])
        if use_token:
            part1_token_pred_list.append(pred_tokens)
            part1_token_gold_list.append(gold_tokens)


    if pred_tokens == gold_tokens == "<internal>":
        # print("pred", item['pred'])
        # print("gold", item['gold'])
        # print("internal_pred", internal_pred)
        # print("external_pred", external_pred)
        # assert False
        if acc(item['pred'], item['gold']) == 1 and acc(internal_pred, item['gold']) == 1 and acc(external_pred, item['gold']) == 0:
            if num_internal >= 20:
                continue
            num_internal += 1
            tmp_dict = {
                "prompt": item['user_prompt'],
                "pred": item['pred'],
                "gold": item['gold'],
                "internal_pred": internal_pred,
                "external_pred": external_pred,
                "audio": item['audio']
            }
            # copy audio to internal_data folder, and use new path in tmp_dict
            tmp_dict['audio'] = f"internal_data/{os.path.basename(item['audio'].split('dev/')[1])}"
            os.makedirs(f"internal_data", exist_ok=True)
            os.system(f"cp {item['audio']} internal_data/")
            with open(f"internal_data.json", "a") as f:
                json.dump(tmp_dict, f)
                f.write("\n")
    elif pred_tokens == gold_tokens == "<external>":
        if acc(item['pred'], item['gold']) == 1 and acc(internal_pred, item['gold']) == 0 and acc(external_pred, item['gold']) == 1:
            if num_external >= 20:
                continue
            num_external += 1
            tmp_dict = {
                "prompt": item['user_prompt'],
                "pred": item['pred'],
                "gold": item['gold'],
                "internal_pred": internal_pred,
                "external_pred": external_pred,
                "audio": item['audio']
            }
            # copy audio to external_data folder, and use new path in tmp_dict
            tmp_dict['audio'] = f"external_data/{os.path.basename(item['audio'].split('dev/')[1])}"
            os.makedirs(f"external_data", exist_ok=True)
            os.system(f"cp {item['audio']} external_data/")
            with open(f"external_data.json", "a") as f:
                json.dump(tmp_dict, f)
                f.write("\n")
    elif pred_tokens == gold_tokens == "<rewrite>":
        if acc(item['pred'], item['gold']) == 1 and acc(internal_pred, item['gold']) == 0 and acc(external_pred, item['gold']) == 0:
            if num_rewrite >= 20:
                continue
            num_rewrite += 1
            tmp_dict = {
                "prompt": item['user_prompt'],
                "pred": item['pred'],
                "gold": item['gold'],
                "internal_pred": internal_pred,
                "external_pred": external_pred,
                "audio": item['audio']
            }
            # copy audio to rewrite_data folder, and use new path in tmp_dict
            tmp_dict['audio'] = f"rewrite_data/{os.path.basename(item['audio'].split('dev/')[1])}"
            os.makedirs(f"rewrite_data", exist_ok=True)
            os.system(f"cp {item['audio']} rewrite_data/")
            with open(f"rewrite_data.json", "a") as f:
                json.dump(tmp_dict, f)
                f.write("\n")

# assert False
# compute precision, recall F1 score of part1, part2, part3 token preds
if use_token:
    # show each label's name, not just 1,2,3,4, name is in token_map
    print("part1")
    print(classification_report(part1_token_gold_list, part1_token_pred_list, digits=digits, labels=list(token_map.values())))
    print("part2")
    print(classification_report(part2_token_gold_list, part2_token_pred_list, digits=digits, labels=list(token_map.values())))
    print("part3")
    print(classification_report(part3_token_gold_list, part3_token_pred_list, digits=digits, labels=list(token_map.values())))
    print("part1 confusion matrix")
    print(confusion_matrix(part1_token_gold_list, part1_token_pred_list, labels=list(token_map.values())))
    print("part2 confusion matrix")
    print(confusion_matrix(part2_token_gold_list, part2_token_pred_list, labels=list(token_map.values())))
    print("part3 confusion matrix")
    print(confusion_matrix(part3_token_gold_list, part3_token_pred_list, labels=list(token_map.values())))
    # plot_confusion_matrix(part1_token_gold_list, part1_token_pred_list, list(token_map.values()), "part1 confusion matrix", f"part1_confusion_matrix_{ckpt}.png")
    # plot_confusion_matrix(part2_token_gold_list, part2_token_pred_list, list(token_map.values()), "part2 confusion matrix", f"part2_confusion_matrix_{ckpt}.png")
    # plot_confusion_matrix(part3_token_gold_list, part3_token_pred_list, list(token_map.values()), "part3 confusion matrix", f"part3_confusion_matrix_{ckpt}.png")
    # plot_confusion_matrix(
    #     part2_token_gold_list, part2_token_pred_list, list(token_map.values()),
    #     title="",
    #     outpath="figs/part2_cm.png", normalize=True
    # )
    plot_confusion_matrix(
        part3_token_gold_list, part3_token_pred_list, list(token_map.values()),
        title="",
        outpath="figs/part3_cm.png", normalize=True
    )
    # plot_confusion_matrix(
    #     part1_token_gold_list, part1_token_pred_list, list(token_map.values()),
    #     title="",
    #     outpath="figs/part1_cm.png", normalize=True
    # )
# part1_f1 = precision_recall_fscore_support(part1_token_gold_list, part1_token_pred_list, average='macro')
# part2_f1 = precision_recall_fscore_support(part2_token_gold_list, part2_token_pred_list, average='macro')
# part3_f1 = precision_recall_fscore_support(part3_token_gold_list, part3_token_pred_list, average='macro')
# print("part1", part1_f1, "part2", part2_f1, "part3", part3_f1)

print(len(data), part1_acc, part2_acc, part3_acc)
# print(part3_data[5], part2_data[5])
# print(part1_acc, part2_acc, part3_acc)
assert len(data) == len(part1_data) + len(part2_data) + len(part3_data)
print(part1_acc / len(part1_data))
print(part2_acc / len(part2_data))
print(part3_acc / len(part3_data))
print("total acc", (part1_acc + part2_acc + part3_acc) / len(data))
