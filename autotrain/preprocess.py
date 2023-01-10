in_path = "/home/fanqi/NLP/CS3602-Final-Project/data/development.json"

import orjson
with open(in_path, "rb") as f:
    data = orjson.loads(f.read())

import jiagu
import enum

def seg(line):
    # return jiagu.seg(line)
    return line.split()

sentences = []
labels = []

constants = {
    "poi名称":   "A",
    "poi修饰":   "B",
    "poi目标":   "C",
    "起点名称":  "D",
    "起点修饰":  "E",
    "起点目标":  "F",
    "终点名称":  "G",
    "终点修饰":  "H",
    "终点目标":  "I",
    "途经点名称":"J",
    "请求类型":  "K",
    "出行方式":  "L",
    "路线偏好":  "M",
    "对象":      "N",
    "操作":      "O",
    "序列号":    "P",
    "页码":      "Q",
    "value":    "R"
}

for entry in data:
    for utt_entry in entry:
        sentence = utt_entry["manual_transcript"]
        seg_result = seg(sentence) # type: list
        slot_result = []
        entry_label = {}
        for mark in utt_entry["semantic"]:
            inform_or_deny = "I" if mark[0] == "inform" else "D"
            slot = constants[mark[1]]
            value = mark[2]
            entry_label[value] = f"{inform_or_deny}{slot}"
        for word in seg_result:
            _marked = False
            for key in entry_label:
                if word in key:
                    slot_result.append(entry_label[key])
                    _marked = True
                    break
            if _marked:
                continue
            slot_result.append("X")
    sentences.append(seg_result)
    labels.append(slot_result)

# Output sentences and labels to csv
import pandas
df = pandas.DataFrame({"sentences": sentences, "labels": labels})
df.to_csv("development.csv", index=False)
