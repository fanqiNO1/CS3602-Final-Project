import sys
import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer
world_path = "/home/fanqi/NLP/CS3602-Final-Project/autotrain-nav-chinese-2795482484"
model = AutoModelForTokenClassification.from_pretrained(world_path)
tokenizer = AutoTokenizer.from_pretrained(world_path)
# inputs = tokenizer("帮我导航到江苏路地铁站谢谢喵", return_tensors="pt")
# src_sentence = "帮我导航到江苏路地铁站谢谢喵"
src_sentence = sys.argv[1]
inputs = tokenizer(src_sentence, return_tensors="pt")

outputs = model(**inputs) # shape: (1, #char+2, 17)
np_o = outputs[0].detach().numpy()

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

labels = []
for i in range(np_o.shape[1]):
    if i == 0 or i == np_o.shape[1] - 1:
        continue
    labels.append(id2label[str(np.argmax(np_o[0][i]))])

for char in src_sentence:
    print("{}: {}".format(char, value2str[labels.pop(0)]))
