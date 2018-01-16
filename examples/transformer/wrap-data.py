import sys
import collections
import itertools

def threshold_vocab(fname, threshold):
    word_counts = collections.Counter()
    with open(fname) as fin:
        for line in fin:
            for token in line.split():
                word_counts[token] += 1

    ok = set()
    for word, count in sorted(word_counts.items()):
        if count >= threshold:
            ok.add(word)
    return ok

def load_vocab_from_file(fname):
    vocab = set()
    fv = open(fname, 'rb')
    for line in fv:
        vocab.add(line.strip())
    return vocab

sfname = sys.argv[3] + "." + sys.argv[1] # '/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/train.en'
tfname = sys.argv[3] + "." + sys.argv[2] #'/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/train.vi'

argc = len(sys.argv)
if argc == 7:
    source_vocab = load_vocab_from_file(sys.argv[6] + "." + sys.argv[1]) # '/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/vocab.en'
    target_vocab = load_vocab_from_file(sys.argv[6] + "." + sys.argv[2]) # '/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/vocab.vi'
elif argc == 8:
    source_vocab = threshold_vocab(sfname, int(sys.argv[6]))
    target_vocab = threshold_vocab(tfname, int(sys.argv[7]))
else: exit()

def process_corpus(sf, tf, of, sv=source_vocab, tv=target_vocab):
    with open(of, 'w') as fout:
        with open(sf) as sin:
            with open(tf) as tin:
                for sline, tline in itertools.izip(sin, tin):
                    print >>fout, '<s>',
                    for token in sline.split():
                        if token in sv:
                            print >>fout, token,
                        else:
                            print >>fout, '<unk>',
                    print >>fout, '</s>', '|||',

                    print >>fout, '<s>',
                    for token in tline.split():
                        if token in tv:
                            print >>fout, token,
                        else:
                            print >>fout, '<unk>',
                    print >>fout, '</s>'

def process_corpus_r(sf, tf, of, sv=source_vocab, tv=target_vocab):
    with open(of, 'w') as fout:
        with open(sf) as sin:
            with open(tf) as tin:
                for sline, tline in itertools.izip(sin, tin):
                    print >>fout, '<s>',
                    for token in tline.split():
                        if token in tv:
                            print >>fout, token,
                        else:
                            print >>fout, '<unk>',
                    print >>fout, '</s>', '|||',

		    print >>fout, '<s>',
                    for token in sline.split():
                        if token in sv:
                            print >>fout, token,
                        else:
                            print >>fout, '<unk>',
                    print >>fout, '</s>'

def process_test(sf, of, vocab):
    with open(of, 'w') as fout:
        with open(sf) as sin:
                for sline in sin:
                    print >>fout, '<s>',
                    for token in sline.split():
                        if token in vocab:
                            print >>fout, token,
                        else:
                            print >>fout, '<unk>',
                    print >>fout, '</s>'

ofname = sys.argv[3] + "." + sys.argv[1] + "-" + sys.argv[2] + ".capped" # '/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/train.en-vi.vcb.capped'

process_corpus(sfname, tfname, ofname) #train (for training)
process_corpus(sys.argv[4] + "." + sys.argv[1], sys.argv[4] + "." + sys.argv[2], sys.argv[4] + "." + sys.argv[1] + "-" + sys.argv[2] + ".capped") # process_corpus('/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/tst2012.en', '/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/tst2012.vi', '/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/tst2012.en-vi.vcb.capped') #dev (for training)
process_test(sys.argv[4] + "." + sys.argv[1], sys.argv[4] + "." + sys.argv[1] + ".capped", source_vocab) # process_test('/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/tst2012.en','/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/tst2012.en.vcb.capped', source_vocab) #dev (for decoding)
process_test(sys.argv[5] + "." + sys.argv[1], sys.argv[5] + "." + sys.argv[1] + ".capped", source_vocab) # process_test('/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/tst2013.en','/home/vhoang2/tools/nmt/nmt/scripts/iwslt15/tst2013.en.vcb.capped', source_vocab) #test (for decoding)


