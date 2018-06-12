## conllreader.py
## Purpose: the reader for CoNLL format in the shared task 2012
## Author: Yangfeng Ji
## Date: Dec. 6, 2016
## Time-stamp: <yangfeng 07/03/2017 14:58:53>

'''
Case 1: (1, -, -, 1)
Case 2: (1, 1)
Case 3: (1)
'''

import copy

def is_number(s):
    try:
        val = int(s)
    except ValueError:
        val = -1
    return val

def find_last(li, elem):
    try:
        idx = len(li) - 1 - li[::-1].index(elem)
    except IndexError:
        idx = None
    return idx

class Token(object):
    def __init__(self, sentidx, wordidx, word,
                 entidx=None, entlen=None):
        self.sentidx = sentidx
        self.wordidx = wordidx
        self.word = word
        self.entidx = entidx
        self.entlen = entlen

class Doc(object):
    def __init__(self, filename, tokens):
        self.tokens = tokens
        self.filename = filename

class BookNLPReader(object):
    def __init__(self):
        self.doc = None
        self.curr_entidx = []

    def read(self, fname):
        self.doc = None
        tokens = []
        with open(fname, 'r') as fin:
            fin.readline()
            for line in fin:
                items = line.rstrip().rsplit()
                try:
                    sidx, widx = int(items[1]), int(items[2])
                    word = items[7].lower()
                except:
                    print items
                    import sys
                    sys.exit()
                word = word.replace("=", "+") # very important
                entidx, entlen = self._get_entidx(items[-1])
                token = Token(sidx, widx, word,
                              entidx, entlen)
                tokens.append(token)
        self.doc = Doc(fname, tokens)
        print "DONE"

    def _get_entidx(self, s):
        """ Get the entity index
        """
        entidx, entlen = 0, 1
        sint = int(s)
        if sint >= 0:
            entidx = sint + 1
            if (len(self.curr_entidx) == 0) or  (entidx != self.curr_entidx[-1]):
                self.curr_entidx = [entidx]
                entlen = 1
            else:
                self.curr_entidx.append(entidx)
                entlen = len(self.curr_entidx)
        else:
            self.curr_entidx = []
        # print entidx, entlen
        return (entidx, entlen)

    def get_doc(self):
        return self.doc

    def get_all_entities(self):
        dct, curr_entidx = {}, []
        prev_mention = None
        prev_entidx = 0
        for tok in self.doc.tokens:
            # print tok.entidx, prev_mention, prev_entidx
            if tok.entidx > 0:
                if tok.entidx == prev_entidx:
                    curr_entidx.append(tok.entidx)
                    prev_mention += (" " + tok.word)
                else:
                    # a new entity mention
                    if prev_entidx > 0:
                        try:
                            dct[prev_entidx].append(prev_mention)
                        except KeyError:
                            dct[prev_entidx] = [prev_mention]
                    prev_mention = tok.word
                    prev_entidx = tok.entidx
            else:
                if prev_entidx > 0:
                    # save prev mention
                    try:
                        dct[prev_entidx].append(prev_mention)
                    except KeyError:
                        dct[prev_entidx] = [prev_mention]
                prev_mention = None
                prev_entidx = 0
        print len(dct)
        for (key, item) in dct.iteritems():
            item = list(set(item))
            dct[key] = item
            print "key = {}, item = {}".format(key, item)
        return dct

    def reorder(self):
        """ Reorder entity index
        """
        ## reorder the entity indices, make sure they start from 1
        doc = self.doc
        # entidxlist = [0]
        # for token in doc.tokens:
        #     if token.entidx not in entidxlist:
        #         entidxlist.append(token.entidx)
        # print entidxlist
        # for (tidx, token) in enumerate(doc.tokens):
        #     idx = entidxlist.index(token.entidx)
        #     token.entidx = idx
        #     doc.tokens[tidx] = token
        ## reindex the mention residual length
        # print "Start reordering ..."
        for tidx in range(1, len(doc.tokens)):
            if (tidx < len(doc.tokens) - 1):
                if (doc.tokens[tidx].entidx > 0) and (doc.tokens[tidx+1].entidx != doc.tokens[tidx].entidx):
                    maxlen = doc.tokens[tidx].entlen
                    for ii in range(maxlen):
                        doc.tokens[tidx-ii].entlen = ii + 1
            elif (tidx == len(doc.tokens) - 1):
                # reach the end of this doc
                if (doc.tokens[tidx].entidx > 0):
                    maxlen = doc.tokens[tidx].entlen
                    for ii in range(maxlen):
                        doc.tokens[tidx-ii].entlen = ii + 1
            else:
                raise ValueError("Unexpected situation")
        return doc


def writefile(doc, outfname):
    fout = open(outfname, 'w')
    prev_sidx = 0
    wc = 0
    # print "To write {} tokens".format(len(doc.tokens))
    for token in doc.tokens:
        if token.sentidx > prev_sidx:
            wc = 0
            fout.write("\n")
        if wc > 0:
            fout.write(" ")
        ## No further entity type splits
        if token.entidx == 0:
            text = "{}|{}|{}|{}".format(token.word, 0, token.entidx, token.entlen)
        else:
            text = "{}|{}|{}|{}".format(token.word, 1, token.entidx, token.entlen)
        fout.write(text)
        wc += 1
        prev_sidx = token.sentidx
    fout.write("\n===\n".format(doc.filename))
    fout.close()

            
def test():
    fname = "../data/dickens.oliver.tokens"
    reader = BookNLPReader()
    reader.read(fname)
    # doc = reader.get_doc()
    # print len(doc.tokens)
    # reader.get_all_entities()
    doc = reader.reorder()
    print len(doc.tokens)
    writefile(doc, fname.replace("tokens", "txt"))
    # doclist = reader.get_doclist()
    # fout = open("tmp.txt", "w")
    # for doc in doclist:
    #    writefile(doc, fout)
    # fout.write("\n")
    # fout.close()


if __name__ == '__main__':
    test()
