from .BaseModel import *


class Snapshot(BaseModel):
    def __init__(self, args, kg):
        super(Snapshot, self).__init__(args, kg)

    def switch_snapshot(self):
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        self.ent_embeddings = ent_embeddings
        self.rel_embeddings = rel_embeddings
        '''reinitialize the embeddings'''
        self.reinit_param()


class TransE(Snapshot):
    def __init__(self, args, kg):
        super(TransE, self).__init__(args, kg)

    def loss(self, head, rel, tail=None, label=None):
        new_loss = self.new_loss(head, rel, tail, label)
        return new_loss