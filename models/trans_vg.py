import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import os

from pytorch_pretrained_bert.modeling import BertModel
from models.SD_predict_noise_pipeline import SDPredictNoisePipeline
from models.unet_feature_extractor import UnetFeatExtractor
from .visual_model.detr import build_detr
from .language_model.bert import build_bert
from .vl_transformer import build_vl_transformer
from utils.box_utils import xywh2xyxy
from omegaconf import OmegaConf
from .util import instantiate_from_config
from transformers.models.clip.modeling_clip import CLIPTextModel
from utils.time_benchmarking import timed
from .stable_diffusion_models import UNetWrapper


def save_unet_feat_by_name(feats, names, root_dir="./unet_feature"):
    feats = feats.cpu()
    for i in range(feats.shape[0]):
        name = names[i].split('.')[0]
        feat = feats[i]
        save_path = os.path.join(root_dir, name+".pth")
        torch.save(feat, save_path)
        print(f"save {name}.pth")

class TransVG(nn.Module):
    def __init__(self, args):
        super(TransVG, self).__init__()
        hidden_dim = args.vl_hidden_dim
        self.hidden_dim = hidden_dim
        divisor = 16 if args.dilation else 32
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len
        # print("before init v model\n\n\n")

        self.visumodel = build_detr(args)
        # print("after init v model\n\n\n")
        # print("before init t model\n\n\n")

        self.textmodel = build_bert(args)
        # print("after init t model\n\n\n")
        self.unet_conv1 = nn.Conv2d(320, hidden_dim, kernel_size=1)
        self.unet_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=8, stride=8)
        if args.use_unet:
            # initialize unet branch
            sd_path = "checkpoints/v1-5-pruned-emaonly.ckpt"
            config = OmegaConf.load('models/v1-inference.yaml')
            config.model.params.ckpt_path = f'{sd_path}'
            sd_model = instantiate_from_config(config.model)
            self.encoder_vq = sd_model.first_stage_model
            self.unet = UNetWrapper(sd_model.model, base_size=512)

            # initialize CLIP
            # print("before load clip in transvg.py\n\n\n")
            self.clip_model = CLIPTextModel.from_pretrained("/home/wangsai/.cache/huggingface/clip-vit-large-patch14/")
            # print("after load clip in transvg.py\n\n\n")
            self.clip_model.cuda()
            self.clip_model = self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False

        
        num_total = self.num_visu_token + self.num_text_token + 1 + 64 if args.use_unet else self.num_visu_token + self.num_text_token + 1
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)
        if args.use_unet:
            self.feat_extractor = UnetFeatExtractor()
            # self.register_buffer('unet_feature_extractor', self.feat_extractor.parameters)
            self.unet_proj_1 = nn.Conv2d(320, self.hidden_dim, kernel_size=1)
            self.unet_bn_1 = nn.BatchNorm2d(self.hidden_dim)
            self.unet_relu_1 = nn.ReLU()
            # 定义第二组卷积层
            self.unet_proj_2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1)
            self.unet_bn_2 = nn.BatchNorm2d(self.hidden_dim)
            self.unet_relu_2 = nn.ReLU()
            # 定义第三组卷积层
            self.unet_proj_3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1)
            self.unet_bn_3 = nn.BatchNorm2d(self.hidden_dim)
            self.unet_relu_3 = nn.ReLU()

        self.vl_transformer = build_vl_transformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)


    def forward(self, img_data, text_data, clip_text=None, raw_img=None, use_unet = 1, img_names=[], unet_feats=[], phrases=[]):
        bs = img_data.tensors.shape[0]

        # visual backbone
        t1 = time.time()
        visu_mask, visu_src = self.visumodel(img_data)
        t2 = time.time()
        # print(f"visumodel cost {t2-t1}s")
        visu_src = self.visu_proj(visu_src) # (N*B)xC

        # language bert
        t1 = time.time()

        text_fea = self.textmodel(text_data)
        t2 = time.time()
        # print(f"textmodel cost {t2-t1}s")

        text_src, text_mask = text_fea.decompose()
        assert text_mask is not None
        text_src = self.text_proj(text_src)
        diff_start_t = time.time()
        # print("diffusion inference start")
        if use_unet:
            # 使用vpd unet进行特征提取
            # clip_text = clip_text.squeeze(1)
            # clip_start_t = time.time()
            # # print("clip inference start")
            # clip_features = self.clip_model(input_ids=clip_text).last_hidden_state
            # clip_end_t = time.time()
            # # print(f"clip inference end, cost {clip_end_t-clip_start_t}s\n")
            # with torch.no_grad():
            #     imgs = raw_img
            #     # 这里应当如何调整维度
            #     # imgs = [F.interpolate(i, size=(512, 512), mode='bilinear', align_corners=True) for i in raw_img]
            #     imgs = []
            #     for_start_t = time.time()
            #     # print("resize start")
            #     for i in raw_img:
            #         t = torchvision.transforms.Resize((512, 512))
            #         resized_img = t(i)
            #        imgs.append(resized_img)
            #     for_end_t = time.time()
            #     # print(f"resize end, cost {for_end_t-for_start_t}s\n")
            #     imgs = torch.stack(imgs)
            #     # imgs = F.interpolate(imgs, size=(512, 512), mode='bilinear', align_corners=True)
            #     vqen_start_t = time.time()
            #     # print("vq_encoder inference start")
            #     latents = self.encoder_vq.encode(imgs).mode().detach()
            #     vqen_end_t = time.time()
            #     # print(f"vq_encoder inference end, cost {vqen_end_t-vqen_start_t}s\n")
            #     t = torch.ones((imgs.shape[0],), device=imgs.device).long()
            #     unet_start_t = time.time()
            #     # print("unet inference start")
            #     outs = self.unet(latents, t, c_crossattn=[clip_features])
            #     unet_end_t = time.time()
            #     # print(f"unet inference end, cost {unet_end_t-unet_start_t}s\n")
            # conv_start_t = time.time()
            # # print("conv start")
            # x_c1, x_c2, x_c3, x_c4 = outs
            # print("use_extracted_feature")




            # 使用diffuser的unet进行特征提取
            imgs = raw_img
            # 这里应当如何调整维度
            # imgs = [F.interpolate(i, size=(512, 512), mode='bilinear', align_corners=True) for i in raw_img]
            imgs = []
            for_start_t = time.time()
            # print("resize start")
            for i in raw_img:
                t = torchvision.transforms.Resize((512, 512))
                resized_img = t(i)
                imgs.append(resized_img)
            for_end_t = time.time()
            # print(f"resize end, cost {for_end_t-for_start_t}s\n")
            imgs = torch.stack(imgs).cuda()
            imgs_device = imgs.device

            t1 = time.time()
            x_c1 = self.feat_extractor(inputs=imgs, prompt=phrases, device=imgs_device)
            t2 = time.time()
            # print(f"diffusion cost {t2-t1}s")

            x_c1 = self.unet_conv1(x_c1)
            x_c1 = self.unet_conv2(x_c1)
            # 存储并使用vpd unet提取的特征

            # x_c1 = torch.stack(unet_feats, dim=0)
            # save_unet_feat_by_name(x_c1, img_names)
            # torch.save(x_c1.cpu(), "x_c1.pth")
            # print("\n\n\nsave c1 success\n\n\n")




            # load_start_t = time.time()
            # # x_c1 = torch.load('x_c1.pth')
            # load_end_t = time.time()
            # # print(f"load feature cost {load_end_t-load_start_t}s")
            # cuda_start_t = time.time()
            # x_c1 = x_c1.cuda()
            # cuda_end_t = time.time()
            # # print(f"to cuda cost {cuda_end_t-cuda_start_t}s")
            # # (b 320 64 64) -> (b 256 64 64) 
            # x_c1 = self.unet_proj_1(x_c1)
            # x_c1 = self.unet_bn_1(x_c1)
            # x_c1 = self.unet_relu_1(x_c1)
            # # (b 256 64 64) -> (b 256 32 32)
            # x_c1 = self.unet_proj_2(x_c1)
            # x_c1 = self.unet_bn_2(x_c1)
            # x_c1 = self.unet_relu_2(x_c1)
            # # (b 256 32 32) -> (b 256 16 16)
            # x_c1 = self.unet_proj_3(x_c1)
            # x_c1 = self.unet_bn_3(x_c1)
            # x_c1 = self.unet_relu_3(x_c1)

            # (b 256 16 16) -> (b 256 256)
            x_c1 = x_c1.flatten(2)
            # (b 256 16 16) -> (256 b 256)
            x_c1 = x_c1.permute(2, 0, 1)
            # conv_end_t = time.time()
            # print(f"conv end, cost {conv_end_t - conv_start_t}s")
            unet_mask = torch.zeros((bs, x_c1.shape[0])).to(x_c1.device).to(torch.bool)
        # diff_end_t = time.time()
        # print(f"diffusion inference end, cost {diff_end_t-diff_start_t}s\n")

        # permute BxLenxC to LenxBxC
        text_src = text_src.permute(1, 0, 2)
        text_mask = text_mask.flatten(1)

        # target regression token
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)
        
        vl_src = torch.cat([tgt_src, text_src, visu_src, x_c1], dim=0) if use_unet else torch.cat([tgt_src, text_src, visu_src], dim=0)
        vl_mask = torch.cat([tgt_mask, text_mask, visu_mask, unet_mask], dim=1) if use_unet else torch.cat([tgt_mask, text_mask, visu_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        t1 = time.time()
        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos) # (1+L+N)xBxC
        t2 = time.time()
        # print(f"vl_transformer cost {t2-t1}s")
        vg_hs = vg_hs[0]

        pred_box = self.bbox_embed(vg_hs).sigmoid()

        return pred_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

