#%%
import bert_helper
from ts.torch_handler.base_handler import BaseHandler
from bert_tokenizer import BertTokenizer
from utils import GloVe

#%%
class MyHandler(BaseHandler):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("vocab.txt")
        self.glove = GloVe("vectors.txt")

    def get_embedding(self, sentence):
        # a_layers = bert_helper.get_layers(sentence, self.tokenizer, self.model)
        # _ , a_vec = bert_helper.get_embeddings(a_layers, 1)
        # return a_vec
        embs, length, _ = bert_helper.get_layers_batch(sentence, self.tokenizer, self.model)
        res = bert_helper.get_embeddings_batch(embs, length, method=1)
        # return list(map(lambda x: [x[0], x[1].tolist()], res))
        glove_embs = [self.glove.get_sentence_emb(i) for i in sentence]
        results = []
        for embedding, gloves in zip(res, glove_embs):
            json = {}
            json['tokens'] = embedding[0]
            json['sentence'] = embedding[1].tolist()
            json['glove'] = gloves.tolist()
            results.append(json)
        return results

    def preprocess(self, req):
        print(req)
        res = map(lambda x: x.get("data"), req)
        res = map(lambda x: x.decode("utf-8"), res)
        return list(res)

    def inference(self, x):
        embs = self.get_embedding(x)
        return embs

    def postprocess(self, preds):
        return [preds]
