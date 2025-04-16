import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from mpl_toolkits.axes_grid1 import ImageGrid
import time
class MultiHeadAttention(nn.Module):
    def __init__(self,hidden_dim=256,num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert (
            self.head_dim * num_heads == hidden_dim
        ), "Hidden dimension must be divisible by number of heads"

        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wo = nn.Linear(hidden_dim, hidden_dim,bias=False)
    def check_spda_inputs(self,x):
        assert x.size(1) ==self.num_heads, "Input tensor must have the same number of heads as the model"
        assert x.size(3) == self.hidden_dim//self.num_heads, "Input tensor must have the same head dimension as the model"
    def scaled_dot_product_attention(
        self, query, key, value, attention_mask=None,key_padding_mask=None
    ):
        self.check_spda_inputs(query)
        self.check_spda_inputs(key)
        self.check_spda_inputs(value)
        d_k = query.size(-1)
        tgt_len = query.size(-2)
        src_len = key.size(-2)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if attention_mask is not None:
            if attention_mask.dim()==2:
                assert attention_mask.size(0) == tgt_len, "Attention mask must have the same length as the target sequence"
                assert attention_mask.size(1) == src_len, "Attention mask must have the same length as the source sequence"
                attention_mask = attention_mask.unsqueeze(0)
                logits=logits+attention_mask
            else:
                raise ValueError("Attention mask must be 2D tensor")
            #logits = logits.masked_fill(attention_mask == 0, -1e9)
       # scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            logits = logits+key_padding_mask
        attention=torch.softmax(logits, dim=-1)
        output=torch.matmul(attention, value)
        #print("Logits range:", logits.min().item(), logits.max().item())
        #print("Attention range:", attention.min().item(), attention.max().item())
        return output,attention
        
    def split_into_heads(self,x,num_heads):
        batch_size, seq_len, hidden_dim = x.size()
        x = x.view(batch_size, seq_len, num_heads, hidden_dim//num_heads)
        return x.transpose(1,2)
    def combine_heads(self,x):
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)
    def forward(self,q,k,v,attention_mask=None,key_padding_mask=None):
        q=self.Wq(q)
        k=self.Wk(k)
        v=self.Wv(v)
        q=self.split_into_heads(q,self.num_heads)
        k=self.split_into_heads(k,self.num_heads)
        v=self.split_into_heads(v,self.num_heads)
        attn_values,attn_weights=self.scaled_dot_product_attention(q,k,v,attention_mask,key_padding_mask)
        grouped=self.combine_heads(attn_values)
        output=self.Wo(grouped)
        self.attention_weights=attn_weights
        return output
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        x=x+ self.pe[:, : x.size(1),:]
        return x
class PositionalWiseFeedForward(nn.Module):
    def __init__(self,d_model:int,d_ff:int):
        super(PositionalWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        #self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
class EncoderBlock(nn.Module):
    def __init__(self,n_dim:int,dropout:float,n_heads:int):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(hidden_dim=n_dim, num_heads=n_heads)
        self.norm1 = nn.LayerNorm(n_dim)
        self.ff = PositionalWiseFeedForward(d_model=n_dim, d_ff=4*n_dim)
        self.norm2 = nn.LayerNorm(n_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, src_padding_mask=None):
        attn_output = self.mha(x, x, x, key_padding_mask=src_padding_mask)
        x = x + self.dropout(self.norm1(attn_output))
        ff_output = self.ff(x)
        x = x + self.dropout(self.norm2(ff_output))
        return x
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size:int,
        n_dim:int,
        dropout:float,
        n_encoder_blocks:int,
        n_heads:int,
    ):
        super(Encoder,self).__init__()
        self.n_dim=n_dim
        self.embedding=nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=n_dim)
        self.positional_encoding=PositionalEncoding(d_model=n_dim,dropout=dropout)
        self.encoder_blocks=nn.ModuleList([EncoderBlock(n_dim=n_dim,dropout=dropout,n_heads=n_heads) for _ in range(n_encoder_blocks)])
    def forward(self,x,padding_mask=None):
        x=self.embedding(x)*math.sqrt(self.n_dim)
        x=self.positional_encoding(x)
        for block in self.encoder_blocks:
            x=block(x,src_padding_mask=padding_mask)
        return x
class DecoderBlock(nn.Module):
    def __init__(self,n_dim:int,dropout:float,n_heads:int):
        super(DecoderBlock,self).__init__()
        self.self_attention=MultiHeadAttention(hidden_dim=n_dim,num_heads=n_heads)
        self.cross_attention=MultiHeadAttention(hidden_dim=n_dim,num_heads=n_heads)
        self.norm1=nn.LayerNorm(n_dim)
        self.norm2=nn.LayerNorm(n_dim)
        self.ff=PositionalWiseFeedForward(n_dim,n_dim)
        self.dropout=nn.Dropout(dropout)
        self.norm3=nn.LayerNorm(n_dim)
    def forward(self,tgt,memory,tgt_mask=None,memory_padding_mask=None,tgt_padding_mask=None):
        masked_att_output = self.self_attention(
            q=tgt, k=tgt, v=tgt, attention_mask=tgt_mask, key_padding_mask=tgt_padding_mask)
        x1 = tgt + self.norm1(masked_att_output)
        cross_att_output = self.cross_attention(
            q=x1, k=memory, v=memory, attention_mask=None, key_padding_mask=memory_padding_mask)
        x2 = x1 + self.norm2(cross_att_output)
        ff_output = self.ff(x2)
        output = x2 + self.norm3(ff_output)
        return output
class Decoder(nn.Module):
    def __init__(self,vocab_size:int,n_dim:int,dropout:float,n_decoder_blocks:int,n_heads:int):
        super(Decoder,self).__init__()
        self.n_dim=n_dim
        self.embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=n_dim,padding_idx=0)
        self.positional_encoding=PositionalEncoding(d_model=n_dim,dropout=dropout)
        self.decoder_blocks=nn.ModuleList([DecoderBlock(n_dim=n_dim,dropout=dropout,n_heads=n_heads) for _ in range(n_decoder_blocks)])
    def forward(self,tgt,memory,tgt_mask=None,tgt_padding_mask=None,memory_padding_mask=None):
        x=self.embedding(tgt)
        x=self.positional_encoding(x)
        for block in self.decoder_blocks:
            x=block(x,memory,tgt_mask=tgt_mask,tgt_padding_mask=tgt_padding_mask,memory_padding_mask=memory_padding_mask)
        return x
class Transformer(nn.Module):
    def __init__(self,**kwargs):
        super(Transformer,self).__init__()
        for k,v in kwargs.items():
            print(f"*{k}: {v}")
        self.vocab_size=kwargs.get('vocab_size')
        self.model_dim=kwargs.get('model_dim')
        self.dropout=kwargs.get('dropout')
        self.n_heads=kwargs.get('n_heads')
        self.n_encoder_layers=kwargs.get('n_encoder_layers')
        self.n_decoder_layers=kwargs.get('n_decoder_layers')
        self.batch_size=kwargs.get('batch_size')
        self.PAD_IDX=kwargs.get('pad_idx',0)
        self.encoder=Encoder(vocab_size=self.vocab_size,n_dim=self.model_dim,dropout=self.dropout,n_encoder_blocks=self.n_encoder_layers,n_heads=self.n_heads)
        self.decoder=Decoder(vocab_size=self.vocab_size,n_dim=self.model_dim,dropout=self.dropout,n_decoder_blocks=self.n_decoder_layers,n_heads=self.n_heads)
        self.fc=nn.Linear(self.model_dim,self.vocab_size)
    
    @staticmethod
    def generate_square_subsequent_mask(sz:int):
        mask = (1-torch.triu(torch.ones(sz, sz), diagonal=1)).bool()
        mask=mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask
    def encode(
        self,
        x:torch.Tensor

    )->torch.Tensor:
        mask=(x==self.PAD_IDX).float()
        encoder_padding_mask=mask.masked_fill(mask==1, float('-inf'))
        encoder_output=self.encoder(x,padding_mask=encoder_padding_mask)
        return encoder_output,encoder_padding_mask
    def decode(
        self,
        tgt:torch.Tensor,
        memory:torch.Tensor,
        memory_padding_mask:torch.Tensor=None
    )->torch.Tensor:
        mask=(tgt==self.PAD_IDX).float()
        tgt_padding_mask=mask.masked_fill(mask==1, float('-inf'))
        decoder_output=self.decoder(
        tgt=tgt,
        memory=memory,
        tgt_mask=self.generate_square_subsequent_mask(tgt.size(1)),
        tgt_padding_mask=tgt_padding_mask,
        memory_padding_mask=memory_padding_mask
        )
        output=self.fc(decoder_output)
        return output
    def forward(
        self,
        x:torch.Tensor,
        y:torch.Tensor,
    )->torch.Tensor:
        encoder_output,encoder_padding_mask=self.encode(x)
        decoder_output=self.decode(tgt=y,
        memory=encoder_output,
        memory_padding_mask=encoder_padding_mask)
        return decoder_output
    def predict(self,x:torch.Tensor,
        sos_idx:int=1,
        eos_idx:int=2,
        max_length:int=None,
    )->torch.Tensor:
        x=torch.cat([torch.tensor([sos_idx]),
        x,
        torch.tensor([eos_idx])]).unsqueeze(0)
        encoder_output,mask=self.transformer.encode(x)
        if not max_length:
            max_length=x.size(1)
        outputs=torch.ones((x.size()[0],max_length)).type_as(x).long()*sos_idx
        for step in range(1,max_length):
            y=outputs[:, :step]
            probs=self.transformer.decode(y,encoder_output)
            output=torch.argmax(probs,dim=-1)
            if output[:,-1].detach().numpy() in (eos_idx,sos_idx):
                break
            outputs[:,step]=output[:,-1]
        return outputs
np.random.seed(0)
def generate_random_string():
    len=np.random.randint(10, 20)
    return "".join([chr(x) for x in np.random.randint(97, 97+26, len)])
class ReverseDataset(Dataset):
    def __init__(self,n_samples,pad_idx,sos_idx,eos_idx):
        super(ReverseDataset,self).__init__()
        self.pad_idx=pad_idx
        self.sos_idx=sos_idx
        self.eos_idx=eos_idx
        self.values=[generate_random_string() for _ in range(n_samples)]
        self.labels=[x[::-1] for x in self.values]
    def __len__(self):
        return len(self.values)
    def __getitem__(self, index):
        return self.text_transform(self.values[index].rstrip("\n")),\
            self.text_transform(self.labels[index].rstrip("\n"))
    def text_transform(self,text):
        return torch.tensor([self.sos_idx]+[ord(z)-97+3 for z in text]+[self.eos_idx],dtype=torch.long)
PAD_IDX=0
SOS_IDX=1
EOS_IDX=2
def train(model,optimizer,loader,loss_fn,epoch):
    model.train()
    losses=0
    acc=0
    history_loss=[]
    history_acc=[]
    with tqdm(loader,position=0,leave=True) as tepoch:
        for x,y in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            optimizer.zero_grad()
            logits=model(x,y[:, :-1])
            loss=loss_fn(logits.contiguous().view(-1,model.vocab_size),y[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            losses+=loss.item()
            preds=logits.argmax(dim=-1)
            masked_pred=preds*(y[:, 1:]!=PAD_IDX)
            accuracy=(masked_pred==y[:, 1:]).float().mean()
            acc+=accuracy.item()
            history_acc.append(accuracy.item())
            history_loss.append(loss.item())
            tepoch.set_postfix(loss=loss.item(), accuracy=100.*accuracy.item())
    return losses/len(list(loader)), acc/len(list(loader)), history_loss, history_acc
def evaluate(model,loader,loss_fn):
    model.eval()
    losses=0
    acc=0
    history_acc=[]
    history_loss=[]
    for x,y in tqdm(loader,position=0,leave=True):
        logits = model(x, y[:, :-1])
        loss = loss_fn(logits.contiguous().view(-1, model.vocab_size), y[:, 1:].contiguous().view(-1))
        losses += loss.item()
        
        preds = logits.argmax(dim=-1)
        masked_pred = preds * (y[:, 1:]!=PAD_IDX)
        accuracy = (masked_pred == y[:, 1:]).float().mean()
        acc += accuracy.item()
        
        history_loss.append(loss.item())
        history_acc.append(accuracy.item())

    return losses / len(list(loader)), acc / len(list(loader)), history_loss, history_acc
def collate_fn(batch):
    src_batch,tgt_batch=[],[]
    for src_sample,tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    src_batch=pad_sequence(src_batch,padding_value=PAD_IDX,batch_first=True)
    tgt_batch=pad_sequence(tgt_batch,padding_value=PAD_IDX,batch_first=True)
    return src_batch,tgt_batch
args={
    "vocab_size": 128,
    "model_dim": 128,
    "dropout": 0.1,
    "n_heads": 4,
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
}
model=Transformer(**args)
train_iter = ReverseDataset(50000, pad_idx=PAD_IDX, sos_idx=SOS_IDX, eos_idx=EOS_IDX)
eval_iter = ReverseDataset(10000, pad_idx=PAD_IDX, sos_idx=SOS_IDX, eos_idx=EOS_IDX)
dataloader_train = DataLoader(train_iter, batch_size=256, collate_fn=collate_fn)
dataloader_val = DataLoader(eval_iter, batch_size=256, collate_fn=collate_fn)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,betas=(0.9,0.98),eps=1e-9)
history={
    "train_loss": [],
    "train_acc": [],
    "eval_loss": [],
    "eval_acc": []
}
print("hello")
for epoch in range(1, 4):
    start_time = time.time()
    train_loss, train_acc, hist_loss, hist_acc = train(model, optimizer, dataloader_train, loss_fn, epoch) 
    history["train_loss"]+=(hist_loss)
    history["train_acc"]+=(hist_acc)
    eval_loss, eval_acc, hist_loss, hist_acc = evaluate(model, dataloader_val, loss_fn)
    history["eval_loss"]+=(hist_loss)
    history["eval_acc"]+=(hist_acc)
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.4f} | Time: {time.time()-start_time:.2f}s")
class Translator(nn.Module):
    def __init__(self,transformer):
        super(Translator,self).__init__()
        self.transformer=transformer
    @staticmethod
    def str_to_tokens(s):
        return [ord(x)-97+3 for x in s]
    @staticmethod
    def tokens_to_str(t):
        return "".join([chr(x+97-3) for x in t])
    def __call__(self,sentence,max_length=None,pad=False):
        x=torch.tensor(self.str_to_tokens(sentence))
        x=torch.cat([torch.tensor([SOS_IDX]),x,torch.tensor([EOS_IDX])]).unsqueeze(0)
        encoder_output,mask=self.transformer.encode(x)
        if not max_length:
            max_length=x.size(1)
        outputs=torch.ones((x.size()[0],max_length)).type_as(x).long()*SOS_IDX
        for step in range(1,max_length):
            y=outputs[:, :step]
            probs=self.transformer.decode(y,encoder_output)
            output=torch.argmax(probs,dim=-1)
            print(f"Knowing {y} we output{output[:,-1]}")
            if output[:,-1].detach().numpy() in (EOS_IDX,SOS_IDX):
                break
            outputs[:,step]=output[:,-1]
        return self.tokens_to_str(outputs[0])
translator=Translator(model)
sentence="zzyuanyi"
out=translator(sentence)
print(f"Input: {out}")
fig=plt.figure(figsize=(10., 10.))
images=model.decoder.decoder_blocks[0].cross_attention.attention_weights[0,...].detach().numpy()
grid=ImageGrid(fig, 111, nrows_ncols=(2,2), axes_pad=0.1)
for ax,im in zip(grid,images):
    ax.imshow(im)
plt.show()