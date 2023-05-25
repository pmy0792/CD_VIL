import torch
import torch.nn as nn

class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.count_e_prompt_selection = dict()
        


        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
            else:
                prompt_pool_shape=(self.num_layers, self.pool_size, self.length, embed_dim)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                    
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean 
            
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None, e_upper= False):
        out = dict()
        
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_key_norm = self.l2_normalize(self.prompt_key, dim=-1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1) # B, C
            
            # print(f"prompt_key_norm.shape:{prompt_key_norm.shape}") #pool, dim
            # print(f"x_embed_norm.shape:{x_embed_norm.shape}") #batch, dim

            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t()) # pool_size, B or Pool_size, #class, B
            similarity = similarity.t() # B, pool_size

            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
            out['similarity'] = similarity

            if self.batchwise_prompt:
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                # Unless dimension is specified, this will be flattend if it is not already 1D.
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                    id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                major_prompt_id = prompt_id[major_idx] # top_k
                # expand to batch. when test time
                idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous() # B, top_k 
                
            
                
            if prompt_mask is not None: # when train time ( or testing e_upper)
                idx = prompt_mask # B, top_k
            '''
                if e_upper:
                    unique_values, counts = torch.unique(idx, return_counts=True)
                    # Loop through the unique values and their counts
                    for value, count in zip(unique_values, counts):
                        # Update the count for the value in the dictionary
                        self.count_e_prompt_selection[value.item()] = count.item() + self.count_e_prompt_selection.get(value.item(), 0)
                    
            else: # when test time
                unique_values, counts = torch.unique(idx, return_counts=True)
                # Loop through the unique values and their counts
                for value, count in zip(unique_values, counts):
                    # Update the count for the value in the dictionary
                    self.count_e_prompt_selection[value.item()] = count.item() + self.count_e_prompt_selection.get(value.item(), 0)
            #print(f"e_idx: {idx[0][0].item()}")
            out['test_e_prompt_selection'] = self.count_e_prompt_selection
            '''
            
            out['prompt_idx'] = idx
            if self.use_prefix_tune_for_e_prompt:
                batched_prompt_raw = self.prompt[:,:,idx]  # num_layers, B, top_k, length, C
                num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                )
            else:
                batched_prompt_raw = self.prompt[:,idx]
                num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, top_k * length, embed_dim
                )

            batched_key_norm = prompt_key_norm[idx] # B, top_k, C

            out['selected_key'] = batched_key_norm
            out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

            # Put pull_constraint loss calculation inside
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar
            
            out['e_reduce_sim'] = reduce_sim
        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)
        
        out['batched_prompt'] = batched_prompt

        return out

class SPrompt(nn.Module):
    def __init__(self, embed_dim = 768,args=None):
        super(SPrompt, self).__init__()
        if args.dataset == "DomainNet":
                self.num_classes = 345
                self.classifier_pool = nn.ModuleList([
                    nn.Linear(embed_dim, self.num_classes, bias=True)
                    for i in range(args.total_sessions)
                ])
        elif args.dataset == "CORe50":
            self.num_classes = 50
            self.classifier_pool = nn.ModuleList([
                nn.Linear(embed_dim, self.num_classes, bias=True)
                for i in range(args.total_sessions)
            ])

        else:
            raise ValueError('Unknown datasets for using S-Prompts: {}.'.format(args["dataset"]))
            
        self.s_prompt_pool = nn.ModuleList([
            nn.Linear(embed_dim, args.s_prompt_length, bias=False)
            for i in range(args.total_sessions) # number of total tasks
        ])
        
    def forward(self, x, task_id=-1, train=False): # return embedding with prepended prompts
        out = dict()
        if train:
            bs=x.shape[0]
            # x: (bs, token_len, embed_dim)
            s_prompt = self.s_prompt_pool[task_id].weight #(s_prompt_length, embed_dim)
            classifier = self.classifier_pool[task_id]
            out['classifier']=classifier
            
            s_prompt=s_prompt.unsqueeze(0).repeat(bs,1,1) # (len, dim) -> (1, len, dim) -> (bs, len, dim)

            #print(f"s_prompt.shape: {s_prompt.shape}")
            #print(f"x.shape:{x.shape}")
            out['x_embed_with_s'] = torch.cat([s_prompt, x], dim=1)
            #print(f"x_embed_with_s.shape: {out['x_embed_with_s'].shape}")
        
        else: #! inference시 모든 classifier에 대해 계산
            bs=x.shape[0]
            x_embed_with_s=[]
            s_prompts=[]
            for idx,instance in enumerate(x):
                s_prompt=self.s_prompt_pool[task_id[idx]].weight # (prompt_len, embed_dim)
                #s_prompt=s_prompt.unsqueeze(0)#.repeat(bs,1,1) # (len, dim) -> (1, len, dim) -> (bs, len, dim)
                s_prompts.append(s_prompt)
               
                #print(f"s_prompt.shape:{s_prompt.shape}")
                #print(f"instance.shape:{instance.shape}") # (token_len, embed_dim)?
                x_embed_with_s.append(torch.cat([s_prompt, instance], dim=0))
            #rint(f"len : {len(x_embed_with_s)}")
            #print(f"shape: {x_embed_with_s[0].shape}")
            x_embed_with_s = torch.stack(x_embed_with_s,dim=0)
            out['x_embed_with_s'] = x_embed_with_s # (bs, len, embed_dim)
            #print(f"x_embed_with_s.shape:{x_embed_with_s.shape}")
        return out
        #else: # inference time: use KNN to select classifier and prompt
            
    
class DPrompt(nn.Module):
    def __init__(self, length=11, embed_dim=768, top_k=1, pool_size=None, prompt_length=10, epochs=10):
        super().__init__()
        self.length=length
        self.embed_dim=embed_dim
        self.top_k=top_k
        self.pool_size=pool_size
        self.prompt_length=prompt_length
        self.epochs=epochs
        prompt_pool_shape = (pool_size, length, embed_dim)
        self.prompt_pool = nn.Parameter(torch.randn(prompt_pool_shape))
        self.batchwise_prompt = True
        nn.init.uniform_(self.prompt_pool,-1,1)
        
        key_shape=(pool_size, embed_dim)
        self.prompt_key=nn.Parameter(torch.randn(key_shape))
        nn.init.uniform_(self.prompt_key,-1,1)
        
        print(f"Created D-Prompt: {prompt_pool_shape}")
        
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
            
    def forward(self, x_embed, prompt_mask=None, cls_features=None, task_id=-1, test=-1,epoch=-1,train_count=[], test_count=[]):
        out = dict()
        if cls_features is None:
            x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
        else:
            x_embed_mean = cls_features
            
        prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C
        
        similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
        if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
        else:
                idx = prompt_mask # B, top_k
        batched_prompt_raw = self.prompt_pool[idx] # B, top_k, length, C
        batch_size, top_k, length, c = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

        out['prompt_idx'] = idx

        # Debugging, return sim as well
        out['prompt_norm'] = prompt_norm
        out['x_embed_norm'] = x_embed_norm
        out['similarity'] = similarity

        # Put pull_constraint loss calculation inside
        batched_key_norm = prompt_norm[idx] # B, top_k, C
        out['selected_key'] = batched_key_norm
        x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
        sim = batched_key_norm * x_embed_norm # B, top_k, C
        reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

        out['d_reduce_sim'] = reduce_sim
        
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
        out['train_prompt_selection']=train_count
        out['test_d_prompt_selection']=test_count
        return out