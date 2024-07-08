# if train_model:
import wandb
import ipdb
import torch
from omegaconf import DictConfig
import hydra
import time
import einops
import random
import torch.nn.functional as F
import torch.optim as optim
st = ipdb.set_trace
import numpy as np
import os
import torch.nn as nn
num_layers = 1
batch_style = 'full'
lr=1e-3 #@param
weight_decay = 1.0 #@param
p=97 #@param
d_model = 176 #@param
d_model = 128 #@param
fn_name = 'add' #@param ['add', 'subtract', 'x2xyy2','rand']
fn_name = '+-' #@param ['add', 'subtract', 'x2xyy2','rand']
fn_name = 'add' #@param ['add', 'subtract', 'x2xyy2','rand']
frac_train = 0.8 #@param
num_epochs = 50000 #@param
save_models = False #@param
save_every = 100 #@param
# Stop training when test loss is <stopping_thresh
stopping_thresh = -1 #@param
seed = 0 #@param
num_layers = 1
batch_style = 'full'
d_vocab = p+1
n_ctx = 3
d_mlp = 4*d_model
num_heads = 4

def plus_minus(a,b):
    if a % 2 == 0:
        c = (a + b) % p
    else:
        c = (a - b) % p
    return c


assert d_model % num_heads == 0
d_head = d_model//num_heads
act_type = 'ReLU' #@param ['ReLU', 'GeLU']
# batch_size = 512
use_ln = False
random_answers = np.random.randint(low=0, high=p, size=(p, p))
fns_dict = {'add': lambda x,y:(x+y)%p, 'subtract': lambda x,y:(x-y)%p, 'x2xyy2':lambda x,y:(x**2+x*y+y**2)%p, 'rand':lambda x,y:random_answers[x][y],'+-': lambda x,y:plus_minus(x,y)}
fn = fns_dict[fn_name]
# st()
root = 'vis_checkpoint'

def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    # Helper function to plot multiple lines
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()

# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_model))

    def forward(self, x):
        return torch.einsum('dbp -> bpd', self.W_E[:, x])

class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_vocab))

    def forward(self, x):
        return (x @ self.W_U)

# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model)/np.sqrt(d_model))

    def forward(self, x):
        return x+self.W_pos[:x.shape[-2]]

# LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon = 1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(torch.ones(d_model))
        self.b_ln = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x

# Attention
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        k = self.hook_k(torch.einsum('ihd,bpd->biph', self.W_K, x))
        q = self.hook_q(torch.einsum('ihd,bpd->biph', self.W_Q, x))
        v = self.hook_v(torch.einsum('ihd,bpd->biph', self.W_V, x))
        attn_scores_pre = torch.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = self.hook_attn(F.softmax(self.hook_attn_pre(attn_scores_masked/np.sqrt(self.d_head)), dim=-1))
        z = self.hook_z(torch.einsum('biph,biqp->biqh', v, attn_matrix))
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = torch.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out

# MLP Layers
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        # self.ln = LayerNorm(d_mlp, model=self.model)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ['ReLU', 'GeLU']

    def forward(self, x):
        x = self.hook_pre(torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
        if self.act_type=='ReLU':
            x = F.relu(x)
        elif self.act_type=='GeLU':
            x = F.gelu(x)
        x = self.hook_post(x)
        x = torch.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x


class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []

    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name

    def add_hook(self, hook, dir='fwd'):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output,
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")

    def forward(self, x):
        return x

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        # self.ln1 = LayerNorm(d_model, model=self.model)
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        # self.ln2 = LayerNorm(d_model, model=self.model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x):
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn((self.hook_resid_pre(x)))))
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp((x))))
        return x



class Transformer(nn.Module):
    def __init__(self, num_layers, d_vocab, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, use_cache=False, use_ln=True, joint=False):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache
        self.joint = joint
        if joint:
            self.modality_embed = nn.Embedding(embedding_dim=1, num_embeddings=2)

        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model=[self]) for i in range(num_layers)])
        # self.ln = LayerNorm(d_model, model=[self])
        self.unembed = Unembed(d_vocab, d_model)
        self.use_ln = use_ln

        for name, module in self.named_modules():
            if type(module)==HookPoint:
                module.give_name(name)

    def forward(self, x, inverse_mapping=False):
        x = self.embed(x)

        if self.joint:
            if inverse_mapping:
                x = x + self.modality_embed(torch.tensor([1]).to(x.device))
            else:
                x = x + self.modality_embed(torch.tensor([0]).to(x.device))        
        
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        # x = self.ln(x)
        x = self.unembed(x)
        return x

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')

    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')

def gen_train_test(frac_train, num, seed=0):
    # Generate train and test split
    pairs = [(i, j, num) for i in range(num) for j in range(num)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train*len(pairs))
    return pairs[:div], pairs[div:]


def cross_entropy_high_precision(logits, labels, l1_loss=0.0):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    
    
    pred_labels = logits.argmax(-1)
    correct = pred_labels == labels
    accuracy = sum(correct).float()/len(correct)
    
    return loss, accuracy

def full_loss(model, data, labels):
    # Take the final position only
    logits= model(data)
    logits = logits[:, -1]
    # st()
    return cross_entropy_high_precision(logits, labels)

def full_loss_inverse(model, data, labels):
    # Take the final position only
    # st()
    label_data = torch.cat([labels[:,None], torch.from_numpy(np.stack(data)).cuda()], 1)
    label_data = label_data[:,[0,1,3,2]]
    input_data = label_data[:,:3]
    labels = label_data[:,3:].squeeze(1)
    logits = model(input_data, inverse_mapping=True)[:, -1]
    return cross_entropy_high_precision(logits, labels)


@hydra.main(config_path="config", config_name="config")
def main(args: DictConfig):    
    os.environ['WANDB_API_KEY'] = '899662853ead8246d39f962194401e222ad8517a'
    args.slurm_id = os.environ.get("SLURM_JOB_ID", None)
    args.project_name = args.project_name + '_vis'
    # st()
    if args.debug:
        wandb.init(project=args.project_name, config=dict(args), mode='disabled')
    else:
        wandb.init(project=args.project_name, config=dict(args))
    # st()
    train, test = gen_train_test(frac_train, p, seed)
    print(len(train), len(test))
    train_labels = torch.tensor([fn(i, j) for i, j, _ in train]).to('cuda')
    test_labels = torch.tensor([fn(i, j) for i, j, _ in test]).to('cuda')
    
    model = Transformer(num_layers=num_layers, d_vocab=d_vocab, d_model=d_model, d_mlp=d_mlp, d_head=d_head, num_heads=num_heads, n_ctx=n_ctx, act_type=act_type, use_cache=False, use_ln=use_ln, joint=args.joint_mode)
    model.to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))
    # run_name = f"grok_{int(time.time())}"
    run_name = wandb.run.name
    print(f'Run name {run_name}')
    if save_models:
        os.mkdir(root/run_name)
        save_dict = {'model':model.state_dict(), 'train_data':train, 'test_data':test}
        torch.save(save_dict, root/run_name/'init.pth')
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        losses = []
        if args.inverse_mode:
            inv_train_loss, inv_train_acc = full_loss_inverse(model, train, train_labels)
            with torch.no_grad():
                inv_test_loss, inv_test_acc = full_loss_inverse(model, test, test_labels)            
            losses.append(inv_train_loss)
            train_loss, train_acc, test_loss, test_acc = (torch.tensor(np.nan), torch.tensor(np.nan), torch.tensor(np.nan), torch.tensor(np.nan))
        elif args.joint_mode:
            inv_train_loss, inv_train_acc = full_loss_inverse(model, train, train_labels)
            with torch.no_grad():
                inv_test_loss, inv_test_acc = full_loss_inverse(model, test, test_labels)            
            
            train_loss, train_acc = full_loss(model, train, train_labels)
            with torch.no_grad():
                test_loss, test_acc = full_loss(model, test, test_labels)
            
            losses.append(inv_train_loss)
            losses.append(train_loss)                
        else:
            train_loss, train_acc = full_loss(model, train, train_labels)
            with torch.no_grad():
                test_loss, test_acc = full_loss(model, test, test_labels)
            losses.append(train_loss)               
            
            inv_train_loss, inv_train_acc, inv_test_loss, inv_test_acc = (torch.tensor(np.nan), torch.tensor(np.nan), torch.tensor(np.nan), torch.tensor(np.nan))
             
        
        train_loss_vis = train_loss.item()
        test_loss_vis = test_loss.item()
        
        train_acc_vis = train_acc.item()    
        test_acc_vis = test_acc.item()
        
        inv_test_acc_vis = inv_test_acc.item()
        # st()        
        vis_dict = {'train_loss_vis': train_loss_vis, 'test_loss_vis': test_loss_vis, 'test_acc_vis': test_acc_vis, 'train_acc_vis':train_acc_vis ,
                    'inv_train_loss': inv_train_loss.item(), 'inv_test_loss': inv_test_loss.item(), 'inv_train_acc': inv_train_acc.item(), 'inv_test_acc': inv_test_acc.item()}
        wandb.log(vis_dict, step=epoch)
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        # st()
        total_loss = sum(losses)/len(losses)
        if epoch%100 == 0: print(f"{epoch}_{train_loss.item():.4f}_{test_loss.item():.4f}")#_{train_acc.item():.4f}_{test_acc.item():.4f}")
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if (epoch%save_every == 0):
            try:
                os.mkdir(f'logs/{root}/{run_name}')
            except Exception:
                pass
            if (test_acc_vis == 1.0 and inv_test_acc_vis == 1.0 and args.joint_mode) or (test_acc_vis == 1.0 and not args.joint_mode):
                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'epoch': epoch,
                }                
                torch.save(save_dict, f"logs/{root}/{run_name}/final.pth")
                print(f"Saved model to logs/{root}/{run_name}/final.pth")                
                break
    # st()
    if not save_models:
        os.mkdir(f'logs/{root}/{run_name}')
    save_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'epoch': epoch,
    }
    torch.save(save_dict, f'logs/{root}/{run_name}/final.pth')
    print(f"Saved model to logs/{root}/{run_name}/final.pth")
    


if __name__ == "__main__":
    main()