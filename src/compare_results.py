import os
import argparse
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", type=str, help="enter the path to the result file")
    parser.add_argument("label_root", type=str, help="enter the root directory which contains the txt label files")
    parser.add_argument("--output", default='../output/', type=str, help="enter the root directory you want to count tumor")
    parser.add_argument("--gather_labels", action='store_true')
    args = parser.parse_args()
    return args


def is_date(s):
    return re.match('\d+/\d+/\d+', s) != None


def gather_labels(label_root, output='../output/'):
    labels = {}
    for root, dirs, files in os.walk(label_root):
        txt_list = []
        txt_exist = False
        for filename in files:
            if '.txt' in filename:
                txt_exist = True
                txt_list.append(filename)
        
        if txt_exist:
            max_len = 0
            target_txt = ''
            for txt_file in txt_list:
                with open(os.path.join(root, txt_file), 'r', encoding='utf-8') as f:
                    num_lines = len(f.readlines())
                    if num_lines > max_len:
                        max_len = num_lines
                        target_txt = os.path.join(root, txt_file)

            with open(target_txt, 'r', encoding='utf-8') as f:
                start = False
                for line in f.readlines():
                    if line.strip() != '':
                        if not start and '---' in line:
                            start = True
                            continue
                        if start:
                            cnt_idx = -1
                            cols = line.strip().split()
                            for i in range(len(cols)):
                                if is_date(cols[i]):
                                    cnt_idx = i - 2
                            if cnt_idx > 0 and cols[cnt_idx].isdigit():
                                sample_name = os.path.split(root)[-1] + '_' + cols[0]
                                # sample_name = cols[0]
                                labels[sample_name] = int(cols[cnt_idx])
                                # if sample_name not in labels:
                                #     labels[sample_name] = []
                                # labels[sample_name].append((os.path.split(root)[-1], int(cols[cnt_idx])))

    if not os.path.exists(output):
        os.makedirs(output)
    with open(os.path.join(output, 'label.json'), 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=4)

    return labels


def compare_results(result_file, label_file, output='../output/'):
    if not os.path.exists(result_file) or not os.path.exists(label_file):
        return None
    
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    samples = []
    for sample_name in results.keys():
        if sample_name in labels:
            samples.append((sample_name, results[sample_name], labels[sample_name]))
        
    print(len(samples))
    names, pred, label = zip(*samples)
    mse = np.mean( np.square( (np.array(pred)-np.array(label)) ) )
    print('mse: ', mse)

    df = pd.DataFrame.from_dict({
        'sample name': names, 
        'label': label, 
        'predict': pred
    })

    if not os.path.exists(output):
        os.makedirs(output)
    df.to_csv(os.path.join(output, 'cmp.csv'))

    


if __name__ == "__main__":
    args = parse_args()
    if args.gather_labels:
        gather_labels(args.label_root, args.output)
    else:
        gather_labels(args.label_root, args.output)
        compare_results(args.result_file, os.path.join(args.output, 'label.json'), args.output)