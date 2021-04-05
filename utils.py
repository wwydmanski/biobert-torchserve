import pandas as pd
from collections import defaultdict
import numpy as np
import string
# import bert_helper


class GloVe:
    _LATIN_SIMILAR = """’'‘ÆÐƎƏƐƔĲŊŒẞÞǷȜæðǝəɛɣĳŋœĸſßþƿȝĄƁÇĐƊĘĦĮƘŁØƠŞȘŢȚŦŲƯY̨Ƴąɓçđɗęħįƙłøơşșţțŧųưy̨ƴÁÀÂÄǍĂĀÃÅǺĄÆǼǢƁĆĊĈČÇĎḌĐƊÐÉÈĖÊËĚĔĒĘẸƎƏƐĠĜǦĞĢƔáàâäǎăāãåǻąæǽǣɓćċĉčçďḍđɗðéèėêëěĕēęẹǝəɛ
                        ġĝǧğģɣĤḤĦIÍÌİÎÏǏĬĪĨĮỊĲĴĶƘĹĻŁĽĿʼNŃN̈ŇÑŅŊÓÒÔÖǑŎŌÕŐỌØǾƠŒĥḥħıíìiîïǐĭīĩįịĳĵķƙĸĺļłľŀŉńn̈ňñņŋóòôöǒŏōõőọøǿơœŔŘŖŚŜŠŞȘṢẞŤŢṬŦÞÚÙÛÜǓŬŪŨŰŮŲ
                        ỤƯẂẀŴẄǷÝỲŶŸȲỸƳŹŻŽẒŕřŗſśŝšşșṣßťţṭŧþúùûüǔŭūũűůųụưẃẁŵẅƿýỳŷÿȳỹƴźżžẓ"""
    WHITE_LIST = string.ascii_letters + string.digits + _LATIN_SIMILAR + ' '
    WHITE_LIST += "'"

    def __init__(self, fname):
        self.embeddings = {}
        with open(fname, "r") as f:
            for line in f:
                self.embeddings[line.split(' ')[0]] = np.asarray(list(map(float, line.strip().split(' ')[1:])))

    def purify(self, x):
        return ''.join(list(filter(lambda x: x in GloVe.WHITE_LIST, x)))
    
    def get_emb(self, word):
        try:
            return self.embeddings[self.purify(word)]
        except KeyError:
            return None
        
    def get_sentence_emb(self, text):
        embeddings = [self.get_emb(i.lower()) for i in text.split()]
        embeddings = [i for i in embeddings if i is not None]
        return np.mean(embeddings, axis=0)


def get_embedding(sentence, tokenizer, bert_model):
    if type(sentence) == str:
        a_layers = bert_helper.get_layers(sentence, tokenizer, bert_model)
        _ , a_vec = bert_helper.get_embeddings(a_layers, 1)
        return a_vec
    else: 
        embs, length, _ = bert_helper.get_layers_batch(sentence, tokenizer, bert_model)
        res = bert_helper.get_embeddings_batch(embs, length, method=1)
        return list(map(lambda x: x[1], res))
