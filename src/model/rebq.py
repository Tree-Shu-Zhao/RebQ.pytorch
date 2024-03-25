import copy

import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from transformers import ViltProcessor

Image.MAX_IMAGE_PIXELS = 1000000000

from .rebq_vilt import PromptedVilt


# Reconstruct before Query(RebQ)
class RebQ(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Load model
        self.vilt_model = PromptedVilt.from_pretrained("dandelin/vilt-b32-mlm")
        self.vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        
        # Add a pooler according to MAP paper
        if cfg.POOLER:
            self.pooler = nn.Sequential(
                nn.Linear(cfg.HIDDEN_SIZE, cfg.HIDDEN_SIZE),
                nn.Tanh(),
            )
        
        if cfg.MLP_CLASSIFIER:
            self.classifier = nn.Sequential(
                nn.Linear(cfg.HIDDEN_SIZE, cfg.HIDDEN_SIZE * 2),
                nn.LayerNorm(cfg.HIDDEN_SIZE * 2),
                nn.GELU(),
                nn.Linear(cfg.HIDDEN_SIZE * 2, cfg.TOTAL_LABELS),
            )
        else:
            self.classifier = nn.Linear(cfg.HIDDEN_SIZE, cfg.TOTAL_LABELS)

        # Expand position embeddings
        if cfg.MAX_LENGTH != self.vilt_model.embeddings.text_embeddings.position_ids.shape[1]:
            logger.info(f"Expanding position embeddings from {self.vilt_model.embeddings.text_embeddings.position_ids.shape[1]} to {cfg.MAX_LENGTH}.")

            self.vilt_model.embeddings.text_embeddings.position_ids = torch.tensor(range(cfg.MAX_LENGTH)).long().view(1, -1)
            pos_embs = torch.nn.functional.interpolate(
                self.vilt_model.embeddings.text_embeddings.position_embeddings.weight.view(1, 1, cfg.position_embedding.DEFAULT_LENGTH, cfg.position_embedding.HIDDEN_SIZE), 
                size=(cfg.MAX_LENGTH, cfg.position_embedding.HIDDEN_SIZE), 
                mode='bilinear'
            ).squeeze()
            self.vilt_model.embeddings.text_embeddings.position_embeddings.weight = nn.Parameter(pos_embs)

        # Setup prompts
        self.memory_prompt_generator = PromptPool(cfg.prompt.memory)
        self.memory_prompt_position = cfg.prompt.memory.POSITION
        self.memory_prompt_layers = cfg.prompt.memory.LAYERS

        self.image_prompt_generator = PromptPool(cfg.prompt.image)
        self.image_prompt_position = cfg.prompt.image.POSITION
        self.image_prompt_layers = cfg.prompt.image.LAYERS

        self.text_prompt_generator = PromptPool(cfg.prompt.text)
        self.text_prompt_position = cfg.prompt.text.POSITION
        self.text_prompt_layers = cfg.prompt.text.LAYERS

        # Freeze parameters
        for name, param in self.named_parameters():
            if 'classifier' in name \
                or "prompt" in name \
                or "pooler" in name \
                or "image_prompt_generator" in name \
                or "text_prompt_generator" in name \
                or "memory_prompt_generator" in name \
                or "memory" in name \
                or "prompt_generator" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
    def forward(self, batch):
        # Step1: Agumentation, in data collaborator
        num_aug = batch["num_aug"]
        original_batch_size = len(batch["images"]) - num_aug

        # Step2: Retrieve memory components
        inputs = batch["inputs"].to(self.vilt_model.device)
        with torch.no_grad():
            queries = self.vilt_model(**inputs).last_hidden_state
        memory_prompts = []
        for i in range(self.memory_prompt_layers):
            layer_prompts = []
            for idx in range(original_batch_size+num_aug):
                if batch["missing_types"][idx] == 0:
                    t_mem_prompts = self.memory_prompt_generator(queries[idx, 0, :].unsqueeze(0), i)
                    i_mem_prompts = self.memory_prompt_generator(queries[idx, inputs.input_ids.size()[-1], :].unsqueeze(0), i)
                    mem_prompts = t_mem_prompts + i_mem_prompts
                elif batch["missing_types"][idx] == 1 or batch["missing_types"][idx] == 3:
                    mem_prompts = self.memory_prompt_generator(queries[idx, inputs.input_ids.size()[-1], :].unsqueeze(0), i)
                elif batch["missing_types"][idx] == 2 or batch["missing_types"][idx] == 4:
                    mem_prompts = self.memory_prompt_generator(queries[idx, 0, :].unsqueeze(0), i)
                layer_prompts.append(mem_prompts)
            layer_prompts = torch.vstack(layer_prompts)
            memory_prompts.append(layer_prompts)
        
        # Step3: Reconstruction
        outputs = self.vilt_model(
            **inputs, 
            memory=memory_prompts,
            memory_prompt_position=self.memory_prompt_position,
            memory_prompt_layers=self.memory_prompt_layers,
        ).last_hidden_state

        # Remove memory prompts
        memory_tokens = outputs[:, 1:1+self.cfg.prompt.memory.LENGTH, :].mean(dim=1)
        outputs = torch.cat((
            outputs[:, :1, :],
            outputs[:, 1+self.cfg.prompt.memory.LENGTH:, :],
        ), dim=1,
        )

        # Split outputs into original and augmented
        original_outputs = outputs[:original_batch_size, :, :]
        augmented_outputs = outputs[original_batch_size:, :, :]

        reconstruction_loss = torch.tensor(0.).to(outputs.device)
        reconstruction_cnt = 0
        for idx in range(num_aug):
            if batch["missing_types"][idx+original_batch_size] == 3:
                reconstruction_loss += ((augmented_outputs[idx, 0, :] - original_outputs[batch["rec_gts"][idx], 0, :].detach())**2).mean()
                reconstruction_cnt += 1
            elif batch["missing_types"][idx+original_batch_size] == 4:
                reconstruction_loss += ((augmented_outputs[idx, inputs.input_ids.size()[-1], :] - original_outputs[batch["rec_gts"][idx], inputs.input_ids.size()[-1], :].detach())**2).mean()
                reconstruction_cnt += 1

        if reconstruction_cnt > 0:
            reconstruction_loss /= reconstruction_cnt
        else:
            reconstruction_loss = None
        
        # Step4: Retrieve vision and language components
        text_queries = []
        image_queries = []
        for idx in range(original_batch_size):
            if batch["missing_types"][idx] == 0:
                text_queries.append(queries[idx, 0, :])
                image_queries.append(queries[idx, inputs.input_ids.size()[-1], :])
            elif batch["missing_types"][idx] == 1:
                text_queries.append(original_outputs[idx, 0, :])
                image_queries.append(queries[idx, inputs.input_ids.size()[-1], :])
            elif batch["missing_types"][idx] == 2:
                text_queries.append(queries[idx, 0, :])
                image_queries.append(original_outputs[idx, inputs.input_ids.size()[-1], :])
            else:
                raise ValueError(f"Cannot find self.cfg.prompt.RECONSTRUCT_TOKEN: {self.cfg.prompt.RECONSTRUCT_TOKEN}.")
        text_queries = torch.vstack(text_queries)
        image_queries = torch.vstack(image_queries)

        language_prompts = [self.text_prompt_generator(text_queries, i) for i in range(self.text_prompt_layers)]
        vision_prompts = [self.image_prompt_generator(image_queries, i) for i in range(self.image_prompt_layers)]
        
        # Step5: Downstream tasks
        # Split inputs
        original_inputs = {key: value[:original_batch_size, ...] for key, value in inputs.items()}

        outputs = self.vilt_model(
            **original_inputs, 
            #**inputs, 
            vision_prompts=vision_prompts,
            vision_prompt_position=self.image_prompt_position,
            vision_prompt_layers=self.image_prompt_layers, 
            language_prompts=language_prompts,
            language_prompt_position=self.text_prompt_position,
            language_prompt_layers=self.text_prompt_layers, 
        ).last_hidden_state

        cls_token = outputs[:, 0, :]
        if self.cfg.POOLER:
            cls_token = self.pooler(cls_token)
        logits = self.classifier(cls_token)
        outputs = {
            "logits": logits,
            "labels": batch["labels"],
        }
        if self.training:
            outputs.update({
                "reconstruction_loss": reconstruction_loss,
            })
        return outputs
    
    def process_task_count(self, task_id):
        self.text_prompt_generator.process_task_count(task_id)
        self.image_prompt_generator.process_task_count(task_id)
        self.memory_prompt_generator.process_task_count(task_id)



class PromptPool(nn.Module):
    def __init__(self, cfg):
        super(PromptPool, self).__init__()
        self.cfg = cfg
        self.task_count = 0
        self.emb_d = cfg.HIDDEN_SIZE
        self.key_d = cfg.KEY_HIDDEN_SIZE
        self.n_tasks = cfg.NUM_TASKS
        self.prompt_layers = cfg.LAYERS

        # prompt basic param
        self.e_pool_size = cfg.POOL_SIZE
        self.e_p_length = cfg.LENGTH
        self.position = cfg.POSITION

        # e prompt init
        for e in range(self.prompt_layers):
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, self.emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)
        
    def process_task_count(self, task_id):
        self.task_count = task_id

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        if self.cfg.REINIT and self.training:
            for e in range(self.prompt_layers):
                K = getattr(self,f'e_k_{e}')
                A = getattr(self,f'e_a_{e}')
                P = getattr(self,f'e_p_{e}')
                k = self.gram_schmidt(K)
                a = self.gram_schmidt(A)
                p = self.gram_schmidt(P)
                setattr(self, f'e_p_{e}',p)
                setattr(self, f'e_k_{e}',k)
                setattr(self, f'e_a_{e}',a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            logger.info('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def forward(self, x_querry, l):

        # e prompts
        e_valid = False
        if l in range(self.prompt_layers):
            e_valid = True

            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
            
            # freeze/control past tasks
            if self.training:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

        # combine prompts for prefix tuning
        if e_valid:
            if self.position == "attention":
                p_return = [Ek, Ev]
            else:
                p_return = torch.hstack((Ek, Ev))
        else:
            p_return = None

        # return
        return p_return


def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p   