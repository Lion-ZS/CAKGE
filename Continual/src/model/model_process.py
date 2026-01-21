from ..utils import *
from ..data_load.data_loader import *
from torch.utils.data import DataLoader

class RetrainBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  
        '''prepare data'''
        self.dataset = TrainDatasetMarginLoss(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=True,
                                      batch_size=int(self.args.batch_size),
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.random_seed)),  
                                      pin_memory=True)

    def process_epoch(self, model, optimizer):
        model.train()
        total_loss = 0.0
        for idx_b, batch in enumerate(self.data_loader):
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = model.loss(bh.to(self.args.device),
                                       br.to(self.args.device),
                                       bt.to(self.args.device),
                                       by.to(self.args.device) if by is not None else by).float()

            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            model.epoch_post_processing(bh.size(0))
        return total_loss

class TrainBatchProcessor():
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.dataset = TrainDatasetMarginLoss(args, kg)
        self.shuffle_mode = True
        if self.args.use_multi_layers:
            self.shuffle_mode = False
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=self.shuffle_mode,
                                      batch_size=int(self.args.batch_size),
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.random_seed)), 
                                      pin_memory=True
                                      ) 
        self.entity_mask_dict = dict() 
        self.relation_mask_dict = dict() 
        self.nodes_sum_mask_weight = None

    def process_epoch(self, model, optimizer):
        model.train()
        total_loss = 0.0
        if self.args.use_multi_layers and self.args.snapshot > 0 and self.args.epoch == 0:
            old_ent_embeddings = model.old_data_ent_embeddings_weight
            all_ent_embedidngs = model.ent_embeddings.weight
            self.entity_mask_dict = dict() 
            entity_mask = [0] * len(all_ent_embedidngs)
            all_entities = set() 
            for i in range(len(old_ent_embeddings)):
                entity_mask[i] = 1
                all_entities.add(i)
            self.entity_mask_dict[0] = torch.Tensor(deepcopy(entity_mask))
            if self.args.using_mask_weight:
                nodes_between_path = self.args.data_path + str(self.args.snapshot) + "/" + "train_nodes_betweenness.txt"
                nodes_between_dict = dict() 
                with open(nodes_between_path, "r", encoding="utf-8") as rf:
                    for line in rf.readlines():
                        line = line.strip()
                        line = line.split("\t")
                        node, v = int(line[0]), float(line[1])
                        nodes_between_dict[node] = v
                nodes_betweenness_mask_weight, new_nodes_betweenness_mask_weight = list(), list()
                for i in range(len(old_ent_embeddings)):
                    nodes_betweenness_mask_weight.append(1)
                nodes_betweenness_mask_weight = torch.tensor(nodes_betweenness_mask_weight, dtype=torch.double)
                for i in range(len(old_ent_embeddings), len(all_ent_embedidngs)):
                    if i not in nodes_between_dict:
                        new_nodes_betweenness_mask_weight.append(0)
                    else:
                        new_nodes_betweenness_mask_weight.append(nodes_between_dict[i])
                new_nodes_betweenness_mask_weight = torch.tensor(new_nodes_betweenness_mask_weight, dtype=torch.double)
                nodes_betweenness_mask_weight = torch.cat([nodes_betweenness_mask_weight, new_nodes_betweenness_mask_weight], dim=-1)
                assert len(nodes_betweenness_mask_weight) == len(all_ent_embedidngs)
                nodes_degree_path = self.args.data_path + str(self.args.snapshot) + "/" + "train_nodes_degree.txt"
                nodes_degree_dict = dict() 
                with open(nodes_degree_path, "r", encoding="utf-8") as rf:
                    for line in rf.readlines():
                        line = line.strip()
                        line = line.split("\t")
                        node, v = int(line[0]), float(line[1])
                        nodes_degree_dict[node] = v
                nodes_degree_mask_weight, new_nodes_degree_mask_weight = list(), list()
                for i in range(len(old_ent_embeddings)):
                    nodes_degree_mask_weight.append(1)
                nodes_degree_mask_weight = torch.tensor(nodes_degree_mask_weight, dtype=torch.double)
                for i in range(len(old_ent_embeddings), len(all_ent_embedidngs)):
                    if i in nodes_degree_dict:
                        new_nodes_degree_mask_weight.append(nodes_between_dict[i])
                    else:
                        new_nodes_degree_mask_weight.append(0)
                new_nodes_degree_mask_weight = torch.tensor(new_nodes_degree_mask_weight, dtype=torch.double)
                new_nodes_degree_mask_weight = F.softmax(new_nodes_degree_mask_weight, dim=-1)
                nodes_degree_mask_weight = torch.cat([nodes_degree_mask_weight, new_nodes_degree_mask_weight], dim=-1)
                assert len(nodes_degree_mask_weight) == len(all_ent_embedidngs)
                self.nodes_sum_mask_weight = nodes_betweenness_mask_weight + nodes_degree_mask_weight
                self.nodes_sum_mask_weight = self.nodes_sum_mask_weight.float()
            if self.args.using_relation_distill:
                old_rel_embeddings = model.old_data_rel_embeddings_weight
                all_rel_embeddings = model.rel_embeddings.weight
                self.relation_mask_dict = dict()
                relation_mask = [0] * len(all_rel_embeddings)
                all_relations = set() 
                for i in range(len(old_rel_embeddings)):
                    relation_mask[i] = 1
                    all_relations.add(i)
                self.relation_mask_dict[0] = torch.Tensor(deepcopy(relation_mask))
            for b_id, batch in enumerate(self.data_loader):
                bh, br, bt, by = batch
                for h in bh:
                    if h not in all_entities:
                        all_entities.add(h)
                        entity_mask[h] = 1
                if self.args.using_relation_distill:
                    for r in br:
                        if r not in all_relations:
                            all_relations.add(r)
                            relation_mask[r] = 1
                for t in bt:
                    if t not in all_entities:
                        all_entities.add(t)
                        entity_mask[t] = 1
                self.entity_mask_dict[b_id + 1] = torch.Tensor(deepcopy(entity_mask))
                if self.args.using_relation_distill:
                    self.relation_mask_dict[b_id + 1] = torch.Tensor(deepcopy(relation_mask))
            value = model.ent_embeddings.weight
            model.register_buffer(f'old_data_ent_embeddings_weight', value.clone().detach())
            if self.args.using_relation_distill:
                value_ = model.rel_embeddings.weight
                model.register_buffer(f'old_data_rel_embeddings_weight', value_.clone().detach())
        if self.args.record:
            loss_save_path = "/data/my_cl_kge/save/" + str(self.args.snapshot) + ".txt"
            with open(loss_save_path, "a", encoding="utf-8") as wf:
                wf.write(str(self.args.epoch))
                wf.write("\t")
        for b_id, batch in enumerate(self.data_loader):
            bh, br, bt, by = batch
            optimizer.zero_grad()
            if self.args.using_all_data:
                batch_loss = model.loss(bh.to(self.args.device),
                                    br.to(self.args.device),
                                    bt.to(self.args.device),
                                    by.to(self.args.device) if by is not None else by).float()
            else:
                batch_loss = model.loss(bh.to(self.args.device),
                                        br.to(self.args.device),
                                        bt.to(self.args.device),
                                        by.to(self.args.device) if by is not None else by,
                                        self.entity_mask_dict[b_id].to(self.args.device) if self.args.use_multi_layers and self.args.snapshot else None,
                                        self.relation_mask_dict[b_id].to(self.args.device) if self.args.use_multi_layers and self.args.snapshot and self.args.using_relation_distill else None,
                                        self.nodes_sum_mask_weight.to(self.args.device) if self.args.use_multi_layers and self.args.snapshot and self.args.using_mask_weight else None
                                        ).float()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            model.epoch_post_processing(bh.size(0))
            if self.args.use_multi_layers and self.args.snapshot > 0:
                old_ent_embeddings = model.old_data_ent_embeddings_weight
                old_len = self.kg.snapshots[self.args.snapshot - 1].num_ent
                value = torch.cat([old_ent_embeddings[:old_len], model.ent_embeddings.weight[old_len:]], dim=0)
                model.register_buffer(f'old_data_ent_embeddings_weight', value.clone().detach())
            if self.args.record:
                with open(loss_save_path, "a", encoding="utf-8") as wf:
                    wf.write(str(batch_loss.item()))
                    wf.write("\t")
        if self.args.record:
            with open(loss_save_path, "a", encoding="utf-8") as wf:
                wf.write("\n")
        return total_loss

class DevBatchProcessor():
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.batch_size = 100
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.random_seed)),
                                      pin_memory=True)

    def process_epoch(self, model):
        model.eval()
        num = 0
        results = {}
        hr2t = self.kg.snapshots[self.args.snapshot].hr2t_all
        for batch in self.data_loader:
            head, relation, tail, label = batch
            head = head.to(self.args.device)
            relation = relation.to(self.args.device)
            tail = tail.to(self.args.device)
            label = label.to(self.args.device) 
            num += len(head)
            stage = "Valid" if self.args.valid else "Test"
            """ Get prediction scores """
            pred = model.predict(head, relation, stage=stage) 
            """ filter: filter: If there is more than one tail in the label, we only think that the tail in this triple is right """
            batch_size_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[batch_size_range, tail] 
            pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred) 
            pred[batch_size_range, tail] = target_pred 
            if self.args.predict_result and stage == "Test":
                logits_sorted, indices_sorted = torch.sort(pred, dim=-1, descending=True)
                predict_result_path = "/data/my_cl_kge/save/predict_result/" + str(self.args.snapshot) + "_" + str(self.args.snapshot_test) + ".txt"
                with open(predict_result_path, "a", encoding="utf-8") as af:
                    batch_num = len(head)
                    for i in range(batch_num):
                        top1 = indices_sorted[i][0]
                        top2 = indices_sorted[i][1]
                        top3 = indices_sorted[i][2]
                        af.write(self.kg.id2entity[head[i].detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2relation[relation[i].detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2entity[tail[i].detach().cpu().item()])
                        af.write("\n")
                        af.write(self.kg.id2entity[top1.detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2entity[top2.detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2entity[top3.detach().cpu().item()])
                        af.write("\n")
                        af.write("----------------------------------------------------------")
                        af.write("\n")
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[batch_size_range, tail]
            ranks = ranks.float() 
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                results[f'hits{k + 1}'] = torch.numel(
                    ranks[ranks <= (k + 1)]
                ) + results.get(f'hits{k + 1}', 0.0)
        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results