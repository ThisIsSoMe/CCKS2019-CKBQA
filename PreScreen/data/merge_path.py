import json
import argparse
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description='Next Hop For KBQA')
    parser.add_argument("--fn_in", default='', type=str)
    parser.add_argument("--fn_multi", default="", type=str)
    parser.add_argument("--fn_out", default="", type=str)

    args = parser.parse_args()
    with open(args.fn_in,'r')as f, open(args.fn_multi,'r')as f2, open(args.fn_out,'w')as fout:
        data = json.load(f)
        multi_data = json.load(f2)
        new_data = []
        for line in data:
            q = line['q']
            paths = line['paths']
            if q in multi_data.keys():
                m = multi_data[q]
            paths.extend(m)
            new_line = {}
            new_line['q'] = q
            new_line['paths'] = paths
            new_data.append(new_line)
        json.dump(new_data, fout, ensure_ascii=False)


