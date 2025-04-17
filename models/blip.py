import warnings
from distutils import dist
from operator import concat

warnings.filterwarnings("ignore")

from models.attention import create_vit
from models.utils import interpolate_pos_embed, load_checkpoint, tie_encoder_decoder_weights
from models.bert import BertConfig, BertModel, BertLMHeadModel, init_tokenizer

import torch
from torch import nn
import torch.nn.functional as F



class BLIP_Base(nn.Module):
    def __init__(self,
                 med_config = "configs/med_config.json",
                 img_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            img_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        self.visual_encoder, vision_width = create_vit(vit,img_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

    def forward(self, image, caption, mode='multimodal'):
        assert mode in ['image', 'text', 'multimodal'], "mode must be image, text or multimodal"
        text = self.tokenizer(caption, return_tensors='pt').to(image.device)

        if mode == 'image':
            image_embeds = self.visual_encoder(image)
            return image_embeds

        elif mode == 'text':
            text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode = '')
            return text_output.last_hidden_state

        elif mode == 'multimodal':
            image_embeds = self.visual_encoder(image) # (B, patch_size ** 2, embed_dim)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            text.input_ids[:,0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, encoder_hidden_states = image_embeds,
                                       encoder_attention_mask=image_atts, return_dict=True)
            return output.last_hidden_state # (B, seq_len, embed_dim)


def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model




class BLIP_Decoder(nn.Module):
    def __init__(self,
                 med_config = 'configs/med_config.json',
                 img_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            img_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, img_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    def forward(self, image, caption):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(
            image.device)

        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100

        decoder_output = self.text_decoder(text.input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           labels=decoder_targets,
                                           return_dict=True,
                                           )
        loss_lm = decoder_output.loss

        return loss_lm

    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeqt(num_beams, dim=0)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        model_kwargs = {'encoder_hidden_states': image_embeds, 'encoder_attention_mask': image_atts}

        prompt = [self.prompt] * image.size(0) # (B,)
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(image.device) # (B, prompt_length)
        input_ids[:, 0] = self.tokenizer.bos_token_id # (B, prompt_length)
        input_ids = input_ids[:, :-1] # (B, prompt_length - 1)

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(input_ids = input_ids,
                                                 max_length= max_length,
                                                 min_length= min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id = self.tokenizer.eos_token_id,
                                                 pad_token_id = self.tokenizer.pad_token_id,
                                                 #TODO: Magic number
                                                 repetition_penalty= 1.1,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids = input_ids,
                                                 max_length= max_length,
                                                 min_length= min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id = self.tokenizer.eos_token_id,
                                                 pad_token_id = self.tokenizer.pad_token_id,
                                                 repetition_penalty= repetition_penalty,
                                                 **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        return captions

def blip_decoder(pretrained='', **kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert (len(msg.missing_keys) == 0)
    return model

class BLIP_Pretrain(nn.Module):
    def __init(self,
               med_config = 'configs/bert_config.json',
               img_size = 224,
               vit = 'base',
               vit_grad_ckpt = False,
               embed_dim = 768,
               queue_size = 57600,
               momentum = 0.995,
               ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            img_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init()
        self.visual_encoder, vision_width = create_vit(vit, img_size, vit_grad_ckpt)

        if vit == 'base':
            checkpoint = torch.hub.load_state_dict_from_url(
                url = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
                map_location = 'cpu', check_hash = True)
            state_dict = checkpoint['model']
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)

        elif vit == 'large':
            from timm.models.helpers import load_custom_pretrained
            from timm.models.vision_transformer import default_cfgs
            load_custom_pretrained(self.visual_encoder,default_cfgs['vit_large_patch16_224_in21k'])

        self.tokenizer = init_tokenizer()
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased', config=encoder_config,
                                                      add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(embed_dim, 2)

        # create momentum encoders
        self.visual_encoder_m, vision_width = create_vit(vit, img_size)
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=encoder_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m]
                            ]
        self.copy_params()

        # create the queue
        self.register_buffer('image_queue', torch.randn(embed_dim, queue_size))
        self.register_buffer('text_queue', torch.randn(embed_dim, queue_size))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # create the decoder
        decoder_config = BertConfig.from_json_file(med_config)
        decoder_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', config=decoder_config)
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        tie_encoder_decoder_weights(self.text_encoder, self.text_decoder.bert, '', '/attention')

    def forward(self, image, caption, alpha):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]), dim=-1) # (B, embed_dim)

        text = self.tokenizer(caption, padding='max_length', truncation=True, return_tensors='pt').to(image.device)
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict=True, mode = 'text')
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]), dim=-1)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()], dim=1)

            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,
                                                return_dict=True, mode = 'text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_t2i+loss_i2t)/2
        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###============== Image-text Matching ===================###
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:,0] = self.tokenizer.enc_token_id

        # forward the positive image-text pair
        bs = image.size(0)
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True)
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i[:,:bs], dim=1) + 1e-4
            weights_t2i.fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t[:,:bs], dim=1) + 1e-4
            weights_i2t.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b],1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask = text_atts_all,
                                       encoder_hidden_states = image_embeds_all,
                                       encoder_attention_mask = image_atts_all,
                                       return_dict=True,
                                       )
        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]], dim=0)
        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2*bs, dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        ##================= LM ========================##
        decoder_input_ids = text.input_ids.clone()
        decoder_input_ids[:,0] = self.tokenizer.bos_token_id
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)

        decoder_output = self.text_decoder(decoder_input_ids,
                                           attention_mask = text.attention_mask,
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,
                                           labels = decoder_targets,
                                           return_dict = True,
                                           )
        loss_lm = decoder_output.loss
        return loss_ita, loss_itm, loss_lm


    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)

        assert self.queue_size % batch_size == 0

        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr


@torch.no_grad()
def concat_all_gather(tensor):
    tensor_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensor_gather, tensor, async_op=False)

    output = torch.cat(tensor_gather, dim=0)
    return output

