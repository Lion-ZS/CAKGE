from .BaseModel import *

class CAKGE(BaseModel):
    def __init__(self, args, kg) -> None:
        super(CAKGE, self).__init__(args, kg)
        self.old_triples_weights = []
        self.num_old_triples = self.args.num_old_triples
        self.num_old_entities = 1000
        self.degree_ent = {}
        self.degree_rel = {}
        self.new_degree_ent = {}
        self.new_degree_rel = {}
        self.alpha_ot = 0.1
        self.alignment_weight =0.5 
        self.lambda_align =0.5 

    def featurewise_ot_align(self, e_old, e_new):
        n = min(e_old.shape[0], e_new.shape[0])
        d = e_old.shape[1]
        ot_term = torch.zeros_like(e_new[:n])
        for j in range(d):
            src_j, _ = torch.sort(e_old[:n, j])
            tgt_j, idx_tgt = torch.sort(e_new[:n, j])
            ot_term[idx_tgt, j] = src_j - tgt_j
        
        return ot_term

    def pre_snapshot(self):
        if self.args.using_mask_weight and self.args.snapshot:
            self.num_new_entity = self.kg.snapshots[self.args.snapshot].num_ent - self.kg.snapshots[self.args.snapshot - 1].num_ent
            self.entity_weight_linear = nn.Linear(self.num_new_entity, self.num_new_entity, bias=False)
            constant_(self.entity_weight_linear.weight, 1e-3)
            self.entity_weight_linear.cuda()

    def store_old_parameters(self):
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(f'old_data_{name}', value.clone().detach())

    def store_previous_old_parameters(self):
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if param.requires_grad:
                value = param.data
                self.register_buffer(
                    f'old_data_{self.args.snapshot}_{name}', value.clone().detach()
                )

    def reply(self):
        self.old_triples_weights = list()
        i_sum = 0
        old_nums = []
        for i in range(0, self.args.snapshot + 1):
            i_sum += i + 1
        for i in range(0, self.args.snapshot + 1):
            old_nums.append((i + 1) * self.args.num_old_triples // i_sum)
        old_nums = old_nums[::-1]
        for i in range(len(old_nums)):
            self.old_triples_weights += list(random.sample(self.kg.snapshots[i].train, old_nums[i]))

    def switch_snapshot(self):
        if self.args.using_multi_embedding_distill == False:
            self.store_old_parameters() 
        else:
            self.store_previous_old_parameters() 
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        
        old_num_ent = self.kg.snapshots[self.args.snapshot].num_ent
        old_num_rel = self.kg.snapshots[self.args.snapshot].num_rel
        new_ent_embeddings[:old_num_ent] = self.ent_embeddings.weight.data
        new_rel_embeddings[:old_num_rel] = self.rel_embeddings.weight.data
        
        if self.alpha_ot > 0 and self.args.snapshot > 0:
            if self.args.using_multi_embedding_distill:
                old_ent_emb = getattr(self, f"old_data_{self.args.snapshot - 1}_ent_embeddings_weight")
            else:
                old_ent_emb = self.old_data_ent_embeddings_weight
            
            n_old = old_ent_emb.shape[0]
            current_ent_emb = new_ent_embeddings[:n_old]
            
            ot_term = self.featurewise_ot_align(old_ent_emb, current_ent_emb)
            
            new_ent_embeddings[:n_old] = current_ent_emb + self.alpha_ot * ot_term
        
        self.ent_embeddings.weight = Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = Parameter(new_rel_embeddings)
        if self.args.using_structure_distill or self.args.using_score_distill or self.args.using_reply:
            self.reply()

    def embedding(self, stage=None):
        if not self.args.use_two_stage or self.args.epoch > self.args.two_stage_epoch_num:
            return self.ent_embeddings.weight, self.rel_embeddings.weight
        else:
            new_ent_embeddings = self.ent_embeddings.weight
            new_rel_embeddings = self.rel_embeddings.weight
            if self.args.snapshot > 0:
                old_ent_embeddings = self.old_data_ent_embeddings_weight
                old_rel_embeddings = self.old_data_rel_embeddings_weight
                old_ent_len = self.kg.snapshots[self.args.snapshot - 1].num_ent
                old_rel_len = self.kg.snapshots[self.args.snapshot - 1].num_rel
                ent_embeddings = torch.cat([old_ent_embeddings[:old_ent_len], new_ent_embeddings[old_ent_len:]])
                rel_embeddings = torch.cat([old_rel_embeddings[:old_rel_len], new_rel_embeddings[old_rel_len:]])
            else:
                ent_embeddings = new_ent_embeddings
                rel_embeddings = new_rel_embeddings
            return ent_embeddings, rel_embeddings

class TransE(CAKGE):
    def __init__(self, args, kg) -> None:
        super(TransE, self).__init__(args, kg)
        self.huber_loss = torch.nn.HuberLoss(reduction='sum')

    def get_TransE_loss(self, head, relation, tail, label):
        return self.new_loss(head, relation, tail, label)

    def get_old_triples(self):
        if isinstance(self.old_triples_weights ,list):
            return self.old_triples_weights
        return list(self.old_triples_weights.keys())

    def structure_loss(self, triples):
        h = [x[0] for x in triples]
        h = torch.LongTensor(h).to(self.args.device)
        t = [x[2] for x in triples]
        t = torch.LongTensor(t).to(self.args.device)
        if self.args.using_multi_embedding_distill:
            old_ent_embeddings = getattr(
                self, f"old_data_{self.args.snapshot - 1}_ent_embeddings_weight"
            )
        else:
            old_ent_embeddings = self.old_data_ent_embeddings_weight
        old_h = torch.index_select(old_ent_embeddings, 0, h)
        old_t = torch.index_select(old_ent_embeddings, 0, t)
        new_h = torch.index_select(self.ent_embeddings.weight, 0, h)
        new_t = torch.index_select(self.ent_embeddings.weight, 0, t)
        loss = self.huber_loss(F.cosine_similarity(old_h, old_t), F.cosine_similarity(new_h, new_t))
        old_h_t = torch.norm(old_h, dim=1) / torch.norm(old_t, dim=1)
        new_h_t = torch.norm(new_h, dim=1) / torch.norm(new_t, dim=1)
        loss += self.huber_loss(old_h_t, new_h_t)
        return loss

    def get_structure_distill_loss(self):
        if self.args.snapshot == 0:
            return 0.0
        triples = self.get_old_triples()
        return self.structure_loss(triples)

    def score_distill_loss(self, head, relation, tail):
        if self.args.using_multi_embedding_distill:
            old_ent_embeddings = getattr(
                self, f"old_data_{self.args.snapshot - 1}_ent_embeddings_weight"
            )
            old_rel_embeddings = getattr(
                self, f"old_data_{self.args.snapshot - 1}_rel_embeddings_weight"
            )
        else:
            old_ent_embeddings = self.old_data_ent_embeddings_weight
            old_rel_embeddings = self.old_data_rel_embeddings_weight
        new_h = torch.index_select(self.ent_embeddings.weight, 0, head)
        new_r = torch.index_select(self.rel_embeddings.weight, 0, relation)
        new_t = torch.index_select(self.ent_embeddings.weight, 0, tail)
        new_score = self.score_fun(new_h, new_r, new_t)
        old_h = torch.index_select(old_ent_embeddings, 0, head)
        old_r = torch.index_select(old_rel_embeddings, 0, relation)
        old_t = torch.index_select(old_ent_embeddings, 0, tail)
        old_score = self.score_fun(old_h, old_r, old_t)
        return self.huber_loss(old_score, new_score)

    def get_score_distill_loss(self):
        if self.args.snapshot == 0:
            return 0.0
        triples = self.get_old_triples()
        triples = torch.LongTensor(triples).to(self.args.device)
        head, relation, tail = triples[:, 0], triples[:, 1], triples[:, 2]
        return self.score_distill_loss(head, relation, tail)

    def corrupt(self, facts):
        ss_id = self.args.snapshot
        label = []
        facts_ = []
        prob = 0.5
        for fact in facts:
            s, r, o = fact[0], fact[1], fact[2]
            neg_s = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
            neg_o = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
            pos_s = np.ones_like(neg_s) * s
            pos_o = np.ones_like(neg_o) * o
            rand_prob = np.random.rand(self.args.neg_ratio)
            sub = np.where(rand_prob > prob, pos_s, neg_s)
            obj = np.where(rand_prob > prob, neg_o, pos_o)
            facts_.append((s, r, o))
            label.append(1)
            for ns, no in zip(sub, obj):
                facts_.append((ns, r, no))
                label.append(-1)
        return facts_, label

    def get_embedding_distillation_loss(self):
        if self.args.snapshot == 0:
            return 0.0
        losses = []
        for name, param in self.named_parameters():
            if name in ["snapshot_weights"]:
                continue
            name = name.replace('.', '_')
            old_data = getattr(self, f'old_data_{name}')
            new_data = param[:old_data.size(0)]
            assert new_data.size(0) == old_data.size(0)
            losses.append(self.huber_loss(old_data, new_data))
        return sum(losses)

    def get_one_layer_loss(self):
        if self.args.snapshot == 0:
            return 0.0
        loss = self.huber_loss(self.old_data_ent_embeddings_weight, self.ent_embeddings.weight)
        return loss

    def get_multi_layer_loss(self, entity_mask, relation_mask, entity_mask_weight):
        if self.args.snapshot == 0 or (self.args.use_two_stage and self.args.epoch < self.args.two_stage_epoch_num):
            return 0.0
        if self.args.use_multi_layers and self.args.using_mask_weight:
            new_entity_mask_weight = self.entity_weight_linear(entity_mask_weight[-self.num_new_entity:])
            entity_mask[-self.num_new_entity:] = entity_mask[-self.num_new_entity:].clone() * new_entity_mask_weight
        if self.args.using_mask_weight == False:
            entity_mask = torch.ones_like(entity_mask) * self.multi_layer_weight
        old_ent_embeddings = self.old_data_ent_embeddings_weight * entity_mask.unsqueeze(1)
        new_ent_embedidngs = self.ent_embeddings.weight * entity_mask.unsqueeze(1)
        loss = self.huber_loss(old_ent_embeddings, new_ent_embedidngs)
        if self.args.using_relation_distill:
            old_rel_embeddings = self.old_data_rel_embeddings_weight * relation_mask.unsqueeze(1)
            new_rel_embeddings = self.rel_embeddings.weight * relation_mask.unsqueeze(1)
            loss += self.huber_loss(old_rel_embeddings, new_rel_embeddings)
        return loss


    def get_multi_embedding_distillation_loss(self):
        if self.args.snapshot == 0:
            return 0.0
        losses = []
        for name, param in self.named_parameters():
            if name == "snapshot_weights":
                continue
            name = name.replace('.', '_')
            for i in range(self.args.snapshot):
                old_data = getattr(self, f'old_data_{i}_{name}')
                new_data = param[:old_data.size(0)]
                assert new_data.size(0) == old_data.size(0)
                losses.append(self.huber_loss(old_data, new_data))
        s_weights = self.snapshot_weights.to(self.args.device).double()
        weights_softmax = F.softmax(s_weights, dim=-1)
        losses = torch.cat([loss.unsqueeze(0) for loss in losses], dim=0)
        loss = torch.dot(losses, weights_softmax)
        print(self.snapshot_weights.grad)
        print(self.snapshot_weights)
        return loss


    def get_reply_loss(self, new_triples, new_labels):
        if self.args.snapshot == 0:
            return 0.0
        old_triples = self.get_old_triples()
        old_triples, old_labels = self.corrupt(old_triples)
        old_triples = torch.LongTensor(old_triples).to(self.args.device)
        old_labels = torch.Tensor(old_labels).to(self.args.device)
        new_triples = torch.cat([new_triples, old_triples], dim=0)
        new_labels = torch.cat([new_labels, old_labels], dim=0)
        head, relation, tail = new_triples[:, 0], new_triples[:, 1], new_triples[:, 2]
        return self.new_loss(head, relation, tail, new_labels)

    def get_contrast_loss(self):
        if self.args.snapshot == 0:
            return 0.0
        old_ent_embeds = self.old_data_ent_embeddings_weight
        old_rel_embeds = self.old_data_rel_embeddings_weight
        new_ent_embeds = self.ent_embeddings.weight
        new_rel_embeds = self.rel_embeddings.weight
        losses = []
        idxs = set()
        for ent in self.new_degree_ent:
            if ent < old_ent_embeds.size(0):
                idxs.add(ent)
        for idx in idxs:
            all_poses = []
            all_poses.append(idx)
            neg_poses = random.sample(range(old_ent_embeds.size(0)), self.args.neg_ratio - 5)
            while idx in neg_poses:
                neg_poses = random.sample(range(old_ent_embeds.size(0)), self.args.neg_ratio - 5)
            all_poses += neg_poses
            student_ent_embeds = new_ent_embeds[all_poses]
            teacher_ent_embeds = old_ent_embeds[all_poses]
            losses.append(infoNCE(student_ent_embeds, teacher_ent_embeds, [0]))
        return sum(losses)

    
    def get_entities_to_update(self):
        if self.args.snapshot == 0:
            return set()
        
        entities_to_update = set()
        current_train = self.kg.snapshots[self.args.snapshot].train
        
        for (h, r, t) in current_train:
            entities_to_update.add(h)
            entities_to_update.add(t)
        
        old_num_ent = self.kg.snapshots[self.args.snapshot - 1].num_ent
        entities_to_update = {e for e in entities_to_update if e < old_num_ent}
        
        return entities_to_update
    
    def compute_semantic_alignment_loss(self):
        if self.args.snapshot == 0:
            return torch.tensor(0.0).to(self.args.device)
        
        entities_to_update = self.get_entities_to_update()
        
        if len(entities_to_update) == 0:
            return torch.tensor(0.0).to(self.args.device)
        
        entity_indices = torch.LongTensor(list(entities_to_update)).to(self.args.device)
        
        if self.args.using_multi_embedding_distill:
            old_ent_emb = getattr(
                self, f"old_data_{self.args.snapshot - 1}_ent_embeddings_weight"
            )
        else:
            old_ent_emb = self.old_data_ent_embeddings_weight
        
        old_emb_subset = torch.index_select(old_ent_emb, 0, entity_indices)
        new_emb_subset = torch.index_select(self.ent_embeddings.weight, 0, entity_indices)
        
        wd_loss = torch.sum((old_emb_subset - new_emb_subset) ** 2)
        
        wd_loss = wd_loss / len(entities_to_update)
        
        return wd_loss
    
    def compute_topological_alignment_loss(self):
        if self.args.snapshot == 0:
            return torch.tensor(0.0).to(self.args.device)
        
        entities_to_update = self.get_entities_to_update()
        
        if len(entities_to_update) == 0:
            return torch.tensor(0.0).to(self.args.device)
        
        entity_indices = torch.LongTensor(list(entities_to_update)).to(self.args.device)
        
        if self.args.using_multi_embedding_distill:
            old_ent_emb = getattr(
                self, f"old_data_{self.args.snapshot - 1}_ent_embeddings_weight"
            )
        else:
            old_ent_emb = self.old_data_ent_embeddings_weight
        
        old_emb_subset = torch.index_select(old_ent_emb, 0, entity_indices)
        new_emb_subset = torch.index_select(self.ent_embeddings.weight, 0, entity_indices)
        
        A_new = F.normalize(new_emb_subset, p=2, dim=1) @ F.normalize(new_emb_subset, p=2, dim=1).T
        
        A_old = F.normalize(old_emb_subset, p=2, dim=1) @ F.normalize(old_emb_subset, p=2, dim=1).T
        
        gwd_loss = torch.sum((A_new - A_old) ** 2)
        
        gwd_loss = gwd_loss / (len(entities_to_update) ** 2)
        
        return gwd_loss
    
    def get_alignment_loss(self):
        if self.args.snapshot == 0:
            return torch.tensor(0.0).to(self.args.device)
        semantic_loss = self.compute_semantic_alignment_loss()
        
        topological_loss = self.compute_topological_alignment_loss()
        
        alignment_loss = self.lambda_align * semantic_loss + (1 - self.lambda_align) * topological_loss
        
        return alignment_loss

    def loss(self, head, relation, tail=None, label=None, entity_mask=None, relation_mask=None, entity_mask_weight=None):
        loss = 0.0
        """ 0. count initial loss """
        if not self.args.using_reply or self.args.snapshot == 0:
            transE_loss = self.get_TransE_loss(head, relation, tail, label) 
            loss = transE_loss
        if self.args.without_multi_layers:
            one_layer_loss = self.get_one_layer_loss() * self.args.embedding_distill_weight
            loss += one_layer_loss
        if self.args.use_multi_layers and (not self.args.without_multi_layers):
            multi_layer_loss = self.get_multi_layer_loss(entity_mask, relation_mask, entity_mask_weight)
            loss += multi_layer_loss * self.args.multi_layer_weight
        
        if self.alignment_weight > 0:
            alignment_loss = self.get_alignment_loss()
            loss += self.alignment_weight * alignment_loss
        return loss

