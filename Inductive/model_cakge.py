import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
from torch_scatter import scatter
SPLIT = '*' * 30
from pre_emb_gnn import PreEmbLayer


class MoEGating(torch.nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=None, temperature=1.0):
        super(MoEGating, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.temperature = temperature
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )
        
        self.expert_scalers = nn.Parameter(torch.ones(num_experts, input_dim))
        nn.init.normal_(self.expert_scalers, mean=1.0, std=0.1)
        
        self.diversity_loss_weight = 0.01
    
    def forward(self, head_emb, rel_emb, tail_emb, training=True):
        context_input = torch.cat([head_emb, rel_emb, tail_emb], dim=-1)
        
        gate_logits = self.gate_network(context_input)
        expert_weights = F.softmax(gate_logits / self.temperature, dim=-1)
        
        batch_size = head_emb.shape[0]
        expert_outputs = []
        
        for i in range(self.num_experts):
            scaled_emb = (1.0 + self.expert_scalers[i].unsqueeze(0)) * head_emb
            expert_outputs.append(scaled_emb)
        
        expert_stack = torch.stack(expert_outputs, dim=1)  
        expert_weights_expanded = expert_weights.unsqueeze(-1)  
        
        fused_emb = torch.sum(expert_weights_expanded * expert_stack, dim=1)
        fused_emb = 0.8 * fused_emb + 0.2 * head_emb  
        
        diversity_loss = 0.0
        if training and self.training:
            avg_weights = expert_weights.mean(dim=0)
            diversity_loss = -torch.sum(avg_weights * torch.log(avg_weights + 1e-8))
            diversity_loss *= self.diversity_loss_weight
        
        return fused_emb, expert_weights, diversity_loss


class OptimalTransportAlignment(torch.nn.Module):
    def __init__(self, feature_dim, alpha=0.05):
        super(OptimalTransportAlignment, self).__init__()
        self.alpha = alpha
        self.feature_dim = feature_dim
        self.warmup_steps = 1000
        self.current_step = 0
        
    def get_adaptive_alpha(self):
        if self.current_step < self.warmup_steps:
            return self.alpha * (self.current_step / self.warmup_steps)
        return self.alpha
        
    def featurewise_ot(self, hidden_old, hidden_new, alpha=None):
        if alpha is None:
            alpha = self.get_adaptive_alpha()
            
        if hidden_old is None or alpha == 0:
            return hidden_new
            
        self.current_step += 1
        n1, d = hidden_old.shape
        n2 = hidden_new.shape[0]
        
        if n2 < n1:
            repeat_factor = (n1 + n2 - 1) // n2
            hidden_new_expanded = hidden_new.repeat(repeat_factor, 1)[:n1]
        else:
            similarity = F.cosine_similarity(
                hidden_old.unsqueeze(1), 
                hidden_new.unsqueeze(0), 
                dim=2
            )
            _, idx = torch.topk(similarity, k=1, dim=1)
            idx = idx.squeeze(1)
            hidden_new_expanded = hidden_new[idx]
        
        aligned_new = hidden_new_expanded.clone()
        for j in range(d):
            h_src_j, _ = torch.sort(hidden_old[:, j])
            h_tgt_j, idx_tgt = torch.sort(aligned_new[:, j])
            update = torch.zeros_like(aligned_new[:, j])
            update[idx_tgt] = alpha * (h_src_j - h_tgt_j)
            aligned_new[:, j] += update
            
        return aligned_new
    
    def align_embeddings(self, old_emb, new_emb):
        if old_emb is None:
            return new_emb
            
        aligned_old = self.featurewise_ot(old_emb, new_emb)
        alpha = self.get_adaptive_alpha()
        return (1 - alpha) * new_emb + alpha * aligned_old


class MultiConditionGNN(torch.nn.Module):
    eps = 1e-6
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, rela_independent, act=lambda x:x, 
                 total_rel=None, use_attn=True, mlp_num=2, num_experts=4, use_moe=True):
        super(MultiConditionGNN, self).__init__()
        self.message_func = 'distmult'
        self.aggregate_func = 'sum'
        self.use_moe = use_moe
        self.num_experts = num_experts

        self.rela_independent = rela_independent
        self.hidden_dim = in_dim
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        self.act = acts['relu']

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.W_attn = nn.Linear(attn_dim, 1, bias=False)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        self.W_1 = nn.Linear(in_dim * 4, 1, bias=True)

        need_rel = total_rel
        self.need_rel = need_rel
        self.rela_embed = nn.Embedding(need_rel, in_dim)
        nn.init.xavier_normal_(self.rela_embed.weight.data)
        self.relation_linear = nn.Linear(in_dim, need_rel * out_dim)

        if self.aggregate_func == 'pna':
            self.agg_linear = nn.Sequential(
                nn.Linear(self.hidden_dim * 12, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim)
            )
        else:
            self.agg_linear = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim)
            )

        self.use_attn = use_attn

        mlp = []
        for x in range(mlp_num):
            mlp.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=True))
            mlp.append(nn.LayerNorm(self.hidden_dim))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(0.1))
        if mlp_num == 0:
            mlp.append(nn.Identity())
        self.mlp = nn.Sequential(*mlp)
        
        if self.use_moe:
            self.moe_gating = MoEGating(in_dim, num_experts, temperature=2.0)
            self.moe_warmup_steps = 500
            self.moe_step = 0
        
        self.ot_alignment = OptimalTransportAlignment(in_dim, alpha=0.05)

    def forward(self, query, q_sub, q_rel, hidden, edges, nodes, rela=None, previous_hidden=None):
        batch_size = hidden.shape[0]
        ent_num = hidden.shape[1]
        dim_size = hidden.shape[2]
        
        id_bat = edges[:, 0]
        id_sub = edges[:, 1]
        id_rel = edges[:, 2]
        id_obj = edges[:, 3]
        
        all_ent_hidden = hidden.flatten(0, 1)
        message_head = torch.index_select(all_ent_hidden, index=id_sub, dim=0)
        
        if rela is None:
            if self.rela_independent:
                relation = self.relation_linear(query).view(batch_size, -1, self.hidden_dim)
                rela_id = id_bat * self.need_rel + id_rel
                message_rela = torch.index_select(relation.flatten(0,1), index=rela_id, dim=0)
            else:
                message_rela = self.rela_embed(id_rel)
        else:
            if type(rela) == torch.nn.modules.linear.Linear:
                assert self.rela_independent
                relation = rela(query).view(batch_size, -1, self.hidden_dim)
                rela_id = id_bat * self.need_rel + id_rel
                message_rela = torch.index_select(relation.flatten(0, 1), index=rela_id, dim=0)
            else:
                message_rela = rela(id_rel)

        message_tail = torch.index_select(all_ent_hidden, index=id_obj, dim=0)
        message_quer = torch.index_select(query, index=id_bat, dim=0)

        diversity_loss = 0.0
        if self.use_moe:
            self.moe_step += 1
            if self.moe_step > self.moe_warmup_steps:
                message_head, expert_weights, diversity_loss = self.moe_gating(
                    message_head, message_rela, message_tail, training=self.training
                )

        if self.message_func == 'transe':
            mess = message_head + message_rela  
        elif self.message_func == 'distmult':
            mess = message_head * message_rela

        if self.use_attn:
            attn_input = self.Ws_attn(message_head) + self.Wr_attn(message_rela) + self.Wqr_attn(message_quer)
            alpha_2 = torch.sigmoid(self.W_attn(self.act(attn_input)))
            message = mess * alpha_2
        else:
            message = mess

        id_mess = id_obj
        agg_dim_size = all_ent_hidden.shape[0]  
        message_agg = scatter(message, index=id_mess, dim=0, dim_size=agg_dim_size, reduce=self.aggregate_func)
        unique_id_mess = torch.unique(id_mess)

        select_nodes_hidden = message_agg[unique_id_mess, :]
        
        residual_input = select_nodes_hidden
        select_nodes_hidden = self.mlp(select_nodes_hidden)
        
        if select_nodes_hidden.shape == residual_input.shape:
            select_nodes_hidden = select_nodes_hidden + residual_input
        
        if (previous_hidden is not None and previous_hidden.numel() > 0 and 
            self.ot_alignment.current_step > self.ot_alignment.warmup_steps // 2):
            if len(previous_hidden.shape) == 3:
                prev_batch_size = previous_hidden.shape[0]
                current_batch_size = batch_size
                
                if prev_batch_size >= current_batch_size:
                    previous_hidden_truncated = previous_hidden[:current_batch_size]
                    previous_hidden_flat = previous_hidden_truncated.flatten(0, 1)
                    max_idx = unique_id_mess.max().item()
                    
                    if previous_hidden_flat.shape[0] > max_idx:
                        prev_select = previous_hidden_flat[unique_id_mess, :]
                        if prev_select.shape == select_nodes_hidden.shape:
                            select_nodes_hidden = self.ot_alignment.align_embeddings(prev_select, select_nodes_hidden)
        
        new_hidden = torch.zeros_like(message_agg)
        new_hidden[unique_id_mess, :] = select_nodes_hidden
        new_hidden = new_hidden.view(batch_size, ent_num, dim_size)
        
        return new_hidden, diversity_loss


class CAKGE(torch.nn.Module):
    def __init__(self, params, loader):
        super(CAKGE, self).__init__()
        self.specific = params.specific
        self.high_way = params.high_way
        self.mlp_num = 2
        
        self.task = params.task
        self.method = params.method
        self.n_layer = params.n_layer
        self.rela_independent = params.rela_independent

        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        
        self.num_experts = getattr(params, 'num_experts', 4)
        self.use_moe = getattr(params, 'use_moe', True)
        self.ot_alpha = getattr(params, 'ot_alpha', 0.05)
        
        self.register_buffer('previous_embeddings', None)
        self.register_buffer('embedding_momentum', None)
        self.momentum_rate = 0.9

        if self.high_way:
            all_rel_num = (2 * self.n_rel + 1) + params.type_num
            self.short_gnn = MultiConditionGNN(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, 
                                        rela_independent=False, total_rel=all_rel_num, use_attn=False, 
                                        mlp_num=0, num_experts=self.num_experts, use_moe=False)  
            self.high_way_rel = 2 * self.n_rel + 1
        else:
            all_rel_num = 2 * self.n_rel + 1
        
        self.layers = []
        for i in range(self.n_layer):
            self.layers.append(MultiConditionGNN(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, 
                                            self.rela_independent, total_rel=all_rel_num, use_attn=True, 
                                            mlp_num=self.mlp_num, num_experts=self.num_experts, use_moe=self.use_moe))
        self.layers = nn.ModuleList(self.layers)
        
        self.rela_embed = nn.Embedding(all_rel_num, self.hidden_dim)
        nn.init.xavier_normal_(self.rela_embed.weight.data)
        self.relation_linear = nn.Linear(self.hidden_dim, all_rel_num * self.hidden_dim)    

        self.W_final = nn.Linear(self.hidden_dim * 2, 1, bias=False)
        mlp = []
        num_mlp_layer = 2
        mlp_hidden = self.hidden_dim * 2
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(mlp_hidden, mlp_hidden))
            mlp.append(nn.LayerNorm(mlp_hidden))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(0.1))
        mlp.append(nn.Linear(mlp_hidden, 1))
        self.mlp = nn.Sequential(*mlp)

        self.role_num = params.type_num
        self.topk = params.type_topk
        self.role_emb = nn.Embedding(self.role_num, self.hidden_dim)
        self.role_layer_n = 6
        self.role_layers = [PreEmbLayer(self.hidden_dim, self.n_rel) for i in range(self.role_layer_n)]
        self.role_layers = nn.ModuleList(self.role_layers)
        
        if self.specific:
            self.query_map_w = nn.Linear(self.hidden_dim, self.hidden_dim * self.hidden_dim)
            self.query_map_b = nn.Linear(self.hidden_dim, self.hidden_dim)

        score = []
        score.append(nn.Linear(self.hidden_dim * 3, self.hidden_dim * 1))
        score.append(nn.LayerNorm(self.hidden_dim * 1))
        score.append(nn.ReLU())
        score.append(nn.Dropout(0.1))
        score.append(nn.Linear(self.hidden_dim * 1, 1))
        self.score = nn.Sequential(*score)
        self.role_classify = nn.Linear(self.hidden_dim, self.role_num - 1)

        if params.dropout is not None:
            self.dropout = nn.Dropout(params.dropout)
        else:
            self.dropout = nn.Dropout(0.1)  

    def update_previous_embeddings(self, embeddings):
        if embeddings.shape[0] >= 10: 
            self.previous_embeddings = embeddings.detach().clone()
            
            if self.embedding_momentum is None:
                self.embedding_momentum = embeddings.detach().clone()
            else:
                if self.embedding_momentum.shape == embeddings.shape:
                    self.embedding_momentum = (self.momentum_rate * self.embedding_momentum + 
                                             (1 - self.momentum_rate) * embeddings.detach())
                else:
                    self.embedding_momentum = embeddings.detach().clone()

    def forward(self, subs, rels, objs, work_mode='train', mode='transductive'):
        device = self.rela_embed.weight.data.device
        batch_size = len(subs)
        n_ent = self.loader.n_ent if work_mode in ['train', 'valid'] else self.loader.n_ent_ind
        q_sub = torch.tensor(subs, dtype=torch.long, device=device)
        q_rel = torch.tensor(rels, dtype=torch.long, device=device)
        q_obj = torch.tensor(objs, dtype=torch.long, device=device)
        if q_sub.shape != q_obj.shape:
            q_obj = None

        query = self.rela_embed(q_rel)
        filter_edges, M_sub = self.loader.get_edges(q_sub, q_rel, q_obj, mode=mode)
        np_filter_edges = filter_edges.detach().cpu().numpy()

        stage1_emb = torch.zeros((batch_size, n_ent, self.hidden_dim), device=device)
        for i in range(self.role_layer_n):
            stage1_emb = self.role_layers[i](query, stage1_emb, q_sub, q_rel, filter_edges, n_ent, relation_linear=self.relation_linear)

        query_emb = self.rela_embed(q_rel).unsqueeze(1).repeat_interleave(n_ent, dim=1)
        head_emb = stage1_emb[torch.arange(batch_size, device=device), q_sub, :].unsqueeze(1).repeat_interleave(n_ent, dim=1)
        stage1_emb = torch.cat([stage1_emb], dim=-1)
        
        if self.specific:
            stage1_emb_w = self.query_map_w(query).view(batch_size, self.hidden_dim, self.hidden_dim)
            stage1_emb_b = self.query_map_b(query).unsqueeze(1)
            stage1_emb_linear = stage1_emb@stage1_emb_w + stage1_emb_b
            stage1_emb = stage1_emb_linear

        nodes_head = torch.cat([torch.arange(batch_size, device=device).unsqueeze(-1), q_sub.unsqueeze(-1)], dim=-1)
        init_hidden = torch.zeros(batch_size, n_ent, self.hidden_dim, device=device)

        def method_base():
            nodes = nodes_head
            return nodes, None

        def method_mstar():
            scores = self.score(torch.cat([stage1_emb, head_emb, query_emb], dim=-1))
            scores = scores.squeeze(-1)
            scores[nodes_head[:, 0], nodes_head[:, 1]] = -10000
            _, argtopk = torch.topk(scores, k=self.topk, dim=-1)
            bid = torch.arange(batch_size, device=device).repeat_interleave(self.topk).unsqueeze(-1)
            eid = argtopk.flatten().unsqueeze(-1)
            entities = torch.cat([bid, eid], dim=-1)
            entity_emb = stage1_emb[entities[:, 0], entities[:, 1], :].view(batch_size, -1, self.hidden_dim)
            logits = self.role_classify(entity_emb)
            r_hard = torch.argmax(logits, dim=-1) + 1
            r = (r_hard.unsqueeze(-1) - logits).detach() + logits
            nodes = torch.cat([nodes_head, entities], dim=0)
            return nodes, entities, r_hard

        def method_random_query():
            if self.topk == 0:
                return method_base()
            pseu_score = torch.rand((batch_size, n_ent), dtype=stage1_emb.dtype, device=device)
            pseu_score[nodes_head[:, 0], nodes_head[:, 1]] = -1000
            topk_simi, topk_ent_id = torch.topk(pseu_score, k=self.topk, dim=-1)
            select_head = topk_ent_id[:, :]
            select_head = select_head.flatten().unsqueeze(1)
            bid = torch.arange(batch_size, device=device).repeat_interleave(self.topk).unsqueeze(1)
            select_head = torch.cat([bid, select_head], dim=-1)
            nodes = torch.cat([nodes_head, select_head], dim=0)
            return nodes, select_head
        
        def method_degree_query():
            one_degree = torch.ones(filter_edges.shape[0], dtype=torch.float, device=device)
            degree = scatter(one_degree, index=filter_edges[:, 0], dim=0, dim_size=n_ent, reduce='sum')
            topk_degree, topk_ent_id = torch.topk(degree, k=self.topk, dim=-1)
            select_head = topk_ent_id.repeat(batch_size).unsqueeze(1)
            bid = torch.arange(batch_size, device=device).repeat_interleave(self.topk).unsqueeze(1)
            select_head = torch.cat([bid, select_head], dim=-1)
            nodes = torch.cat([nodes_head, select_head], dim=0)
            return nodes, select_head
        
        if self.method == "mstar":
            nodes, entities, r_type = method_mstar()
        elif self.method == "None":
            nodes, entities  = method_base()
        elif self.method == 'random_query':
            nodes, entities  = method_random_query()
            r_type = torch.ones(entities.shape[0], dtype=torch.long, device=entities.device)
        elif self.method == 'degree_query':
            nodes, entities  = method_degree_query()
            r_type = torch.ones(entities.shape[0], dtype=torch.long, device=entities.device)
        else:
            assert False

        def reidx(batch_id, node_id):
            return batch_id * n_ent + node_id
        
        init_hidden = torch.zeros(batch_size, n_ent, self.hidden_dim, device=device)
        if self.high_way:
            init_hidden = torch.zeros(batch_size, n_ent, self.hidden_dim, device=device)
            bid = entities[:, [0]]
            high_way_rel = torch.ones_like(bid) * self.high_way_rel
            high_way_rel = high_way_rel -1 + r_type.flatten().unsqueeze(-1)
            high_way_edges = torch.cat([bid, q_sub[bid], high_way_rel, entities[:, [1]]], dim=1)
            high_way_edges[:, 1] = reidx(high_way_edges[:, 0], high_way_edges[:, 1])
            high_way_edges[:, 3] = reidx(high_way_edges[:, 0], high_way_edges[:, 3])
            nodes_heads_emb = self.role_emb(torch.ones(1, dtype=torch.long, device=device) * 0)

            init_hidden[nodes_head[:, 0], nodes_head[:, 1], :] = nodes_heads_emb
            init_hidden, _ = self.short_gnn(query, q_sub, q_rel, init_hidden, high_way_edges, nodes, 
                                         rela=self.rela_embed, previous_hidden=None)
            init_hidden[nodes_head[:, 0], nodes_head[:, 1], :] = nodes_heads_emb
        hidden = init_hidden

        total_diversity_loss = 0.0
        for layer_id in range(self.n_layer):
            next_layer_nodes, selected_edges = self.loader.get_next_layer_nodes_edges(nodes, n_ent, M_sub, np_filter_edges)
            nodes = next_layer_nodes

            selected_edges[:, 1] = reidx(selected_edges[:, 0], selected_edges[:, 1])
            selected_edges[:, 3] = reidx(selected_edges[:, 0], selected_edges[:, 3])
            curr_edges = selected_edges
            
            rela = self.relation_linear if self.rela_independent else self.rela_embed
            
            layer_previous_hidden = (self.previous_embeddings if layer_id == 0 and work_mode == 'train' 
                                   else None)
            new_hidden, diversity_loss = self.layers[layer_id](query, q_sub, q_rel, hidden, curr_edges, nodes, 
                                                             rela, previous_hidden=layer_previous_hidden)
            hidden = new_hidden
            hidden = self.dropout(hidden)
            total_diversity_loss += diversity_loss

        if work_mode == 'train':
            self.update_previous_embeddings(hidden)

        scores_all = torch.zeros((batch_size, n_ent), device=device)
        visited = torch.zeros((batch_size, n_ent), dtype=torch.bool, device=device)
        visited[nodes[:, 0], nodes[:, 1]] = 1
        hidden = torch.cat([hidden, head_emb], dim=-1)
        scores_all = self.mlp(hidden).squeeze(-1)

        return scores_all, visited, total_diversity_loss