import orjson

def read_json(path):
    with open(path, 'rb') as f:
        return orjson.loads(f.read())

def metrics(data):
    metric_all_correct = []
    metric_item_correct = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            utt = data[i][j]
            gt = utt["semantic"]
            pred = utt["pred"]
            gtmap = {}
            for gtitem in gt:
                key = f"{gtitem[0]}/{gtitem[1]}"
                if key in gtmap:
                    gtmap[key].add(gtitem[2])
                else:
                    gtmap[key] = set([gtitem[2]])
            predmap = {}
            for preditem in pred:
                key = f"{preditem[0]}/{preditem[1]}"
                if key in predmap:
                    predmap[key].add(preditem[2])
                else:
                    predmap[key] = set([preditem[2]])
            if all([key in predmap and predmap[key] == gtmap[key] for key in gtmap]):
                metric_all_correct.append(1)
            else:
                metric_all_correct.append(0)
            for key in gtmap:
                if key in predmap:
                    if predmap[key] == gtmap[key]:
                        metric_item_correct.append(1)
                else:
                    metric_item_correct.append(0)
    return sum(metric_all_correct) / len(metric_all_correct), sum(metric_item_correct) / len(metric_item_correct)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    path = args.path
    data = read_json(path)
    m1, m2 = metrics(data)
    print("Accuracy: {:.4f}% / {:.4f}%".format(m1 * 100, m2 * 100))
