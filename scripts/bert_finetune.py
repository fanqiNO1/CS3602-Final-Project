import orjson
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer
world_path = "/home/fanqi/NLP/CS3602-Final-Project/autotrain-nav-chinese-2795482484"
model = AutoModelForTokenClassification.from_pretrained(world_path)
tokenizer = AutoTokenizer.from_pretrained(world_path)

id2label = {
    "0": "DN",
    "1": "DO",
    "2": "IA",
    "3": "IB",
    "4": "IC",
    "5": "ID",
    "6": "IG",
    "7": "IH",
    "8": "II",
    "9": "IJ",
    "10": "IK",
    "11": "IM",
    "12": "IN",
    "13": "IO",
    "14": "IP",
    "15": "IQ",
    "16": "X"
}

value2str = {
    "IA": "inform/poi名称",
    "IB": "inform/poi修饰",
    "IC": "inform/poi目标",
    "ID": "inform/起点名称",
    "IE": "inform/起点修饰",
    "IF": "inform/起点目标",
    "IG": "inform/终点名称",
    "IH": "inform/终点修饰",
    "II": "inform/终点目标",
    "IJ": "inform/途经点名称",
    "IK": "inform/请求类型",
    "IL": "inform/出行方式",
    "IM": "inform/路线偏好",
    "IN": "inform/对象",
    "IO": "inform/操作",
    "IP": "inform/序列号",
    "IQ": "inform/页码",
    "IR": "inform/value",
    "DN": "deny/对象",
    "DO": "deny/操作",
    "X": "none"
}

value2str = {
    "IA": ["inform", "poi名称"],
    "IB": ["inform", "poi修饰"],
    "IC": ["inform", "poi目标"],
    "ID": ["inform", "起点名称"],
    "IE": ["inform", "起点修饰"],
    "IF": ["inform", "起点目标"],
    "IG": ["inform", "终点名称"],
    "IH": ["inform", "终点修饰"],
    "II": ["inform", "终点目标"],
    "IJ": ["inform", "途经点名称"],
    "IK": ["inform", "请求类型"],
    "IL": ["inform", "出行方式"],
    "IM": ["inform", "路线偏好"],
    "IN": ["inform", "对象"],
    "IO": ["inform", "操作"],
    "IP": ["inform", "序列号"],
    "IQ": ["inform", "页码"],
    "IR": ["inform", "value"],
    "DN": ["deny", "对象"],
    "DO": ["deny", "操作"],
    "X": None
}

def inference(src_sentence):
    inputs = tokenizer(src_sentence, return_tensors="pt")
    outputs = model(**inputs) # shape: (1, #char+2, 17)
    np_o = outputs[0].detach().numpy()
    labels = []
    for i in range(np_o.shape[1]):
        if i == 0 or i == np_o.shape[1] - 1:
            continue
        labels.append(id2label[str(np.argmax(np_o[0][i]))])
    # labels: ["IO", "IO", "X", "IG", "IG", "IG", "IG"]

    result_str = []
    cached_lbl = "X"
    cached_word = ""
    for i in range(len(labels)):
        cur_lbl = labels[i]
        if cur_lbl == "X":
            # If there is a cached word, write it to result
            if cached_word != "":
                result_str.append([
                    value2str[cached_lbl][0],
                    value2str[cached_lbl][1],
                    cached_word
                ])
                # Reset cache
                cached_lbl = "X"
                cached_word = ""
        else: # There is a label other than "X"
            if cached_lbl == cur_lbl:
                # If the label is the same as the cached label, add the char to the cache
                cached_word += src_sentence[i]
            else:
                # This is a new label. If there is a cached word, write it to result
                if cached_word != "":
                    result_str.append([
                        value2str[cached_lbl][0],
                        value2str[cached_lbl][1],
                        cached_word
                    ])
                # Reset cache
                cached_lbl = cur_lbl
                cached_word = src_sentence[i]
    # If there is a cached word, write it to result
    if cached_word != "":
        result_str.append([
            value2str[cached_lbl][0],
            value2str[cached_lbl][1],
            cached_word
        ])
    return result_str

def handle_file(file_path, output_path):
    with open(file_path, "rb") as f:
        data = orjson.loads(f.read())
    for i in tqdm(range(len(data))):
        for j in range(len(data[i])):
            input_sentence = data[i][j]["asr_1best"]
            result_str = inference(input_sentence)
            data[i][j]["pred"] = result_str
    with open(output_path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="Path to the test data (JSON). Preferrably absolute path.")
    parser.add_argument("-o", type=str, dest="output_path", help="Path to the output file (JSON).")
    args = parser.parse_args()
    handle_file(args.file_path, args.output_path)
