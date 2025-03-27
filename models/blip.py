import warnings
warnings.filterwarnings("ignore")

from models.attention import create_vit
from models.utils import interpolate_pos_embed, load_checkpoint
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




