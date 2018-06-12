## process.py
## Author: Yangfeng Ji
## Time-stamp: <yangfeng 09/18/2017 23:11:38>

from collections import defaultdict
from operator import itemgetter

def build_vocab(fname, thresh=2):
    counts = defaultdict(int)
    with open(fname) as fin:
        for line in fin:
            items = line.strip().rsplit()
            for item in items:
                strs = item.split("|")
                counts[strs[0]] += 1
    # sorted
    dct = {}
    sorted_counts = sorted(counts.items(), key=itemgetter(1), reverse=True)
    for item in sorted_counts:
        if item[1] >= thresh:
            dct[item[0]] = len(dct)
        else:
            break
    print "len(dct) = ", len(dct)
    return dct


def filter_doc(infname, outfname, dct, thresh=20):
    fout = open(outfname, 'w')
    bcfout = open(outfname + ".bc", "w")
    fin = open(infname, 'r')
    lc, didx = 0, 0
    for line in fin:
        if line.startswith("==="):
            continue
        items = line.strip().rsplit()
        tokens = []
        for (idx, item) in enumerate(items):
            strs = item.split("|")
            try:
                dct[strs[0]]
            except KeyError:
                strs[0] = "UNK"
            tokens.append(strs[0])
            items[idx] = "|".join(strs)
        newline = " ".join(items)
        lc += 1
        fout.write(newline + "\n")
        bcfout.write(" ".join(tokens) + "\n")
        # print "lc = ", lc
        if lc % thresh == 0:
            # print "doc boundary"
            didx += 1
            fout.write("=== {} {}\n".format(infname, didx))
    if lc % thresh != 0:
        didx += 1
        fout.write("=== {} {}\n".format(infname, didx))
    fout.close()
    bcfout.close()
    fin.close()


def main():
    infname = "../data/dickens.oliver.txt"
    outfname = "../data/trn-oliver.txt"
    dct = build_vocab(infname)
    filter_doc(infname, outfname, dct, thresh=50)


if __name__ == '__main__':
    main()
