import os.path as osp
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata
from torchvision.transforms import Compose
from backbones.resnet import ResNet9 
from data.dataset import DataSet
from data.transform import BaseTransform, RandomRotate, RandomBrightness, RandomBlur
from data.sampler import TripletSampler, InferenceSampler
from losses.triplet import TripletLoss
from losses.ce import CrossEntropyLoss
from losses.loss_aggregator import LossAggregator
from utils.common import ts2np, Odict, mkdir
from utils.msg_manager import msg_mgr
from models.modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks

class Baseline(nn.Module):

    def __init__(self, cfgs):
        super(Baseline, self).__init__()

        self.build_network(cfgs['model_cfg'])
        self.init_parameters()
        self.init_loader(cfgs['data_cfg'])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.iteration = 0
        self.msg_mgr = msg_mgr
        self.with_test = cfgs['with_test']
        self.restore_ckpt_strict = cfgs['restore_ckpt_strict']
        self.restore_hint = cfgs['restore_hint']
        self.optimizer_reset = cfgs['optimizer_reset']
        self.scheduler_reset = cfgs['scheduler_reset']
        self.log_iter = cfgs['log_iter']
        self.save_iter = cfgs['save_iter']
        self.total_iter = cfgs['total_iter']
        self.save_path = cfgs['save_path']
        self.save_name = cfgs['save_name']
        self.msg_mgr.init_manager(save_path=self.save_path, log_to_file=True, log_iter=self.log_iter)

        self.loss_aggregator = LossAggregator(cfgs['loss_cfg'])
        self.optimizer = optim.SGD(self.parameters(), **cfgs['optimizer_cfg'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, **cfgs['scheduler_cfg'])

        if self.restore_hint != 0:
            self.resume_ckpt(self.restore_hint)

        self.msg_mgr.log_info(cfgs)

    def build_network(self, model_cfg):
        self.Backbone = ResNet9(**model_cfg['backbone_cfg'])
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.HPP = HorizontalPoolingPyramid(model_cfg['bin_num'])

    def init_loader(self, data_cfg):
        self.root_dir = data_cfg['root_dir']
        self.train_path = data_cfg['train_path']
        self.test_path = data_cfg['test_path']
        self.meta_path = data_cfg['meta_path']

        self.train_dataset = DataSet(csv_path=self.train_path, root_dir=self.root_dir)
        self.test_dataset = DataSet(csv_path=[self.train_path, self.test_path], root_dir=self.root_dir)
        self.meta_dataset = DataSet(csv_path=self.meta_path, root_dir=self.root_dir)
        
        self.train_sampler = TripletSampler(dataset=self.train_dataset, **data_cfg['train_sampler'])
        self.test_sampler = InferenceSampler(dataset=self.test_dataset, **data_cfg['test_sampler'])
        self.meta_sampler = InferenceSampler(dataset=self.meta_dataset, **data_cfg['test_sampler'])
        
        self.train_transform = Compose([
            RandomBrightness(**data_cfg['RandomBrightness']),
            RandomBlur(**data_cfg['RandomBlur']),
            RandomRotate(**data_cfg['RandomRotate']),
            BaseTransform()])
        self.test_transform = Compose([BaseTransform()])
        self.meta_transform = Compose([BaseTransform()])
        
        self.train_loader = tordata.DataLoader(
            dataset=self.train_dataset,
            batch_sampler=self.train_sampler,
            num_workers=1
        )
        self.test_loader = tordata.DataLoader(
            dataset=self.test_dataset,
            batch_sampler=self.test_sampler,
            num_workers=1
        )
        self.meta_loader = tordata.DataLoader(
            dataset=self.meta_dataset,
            batch_sampler=self.meta_sampler,
            num_workers=1
        )

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def inputs_pretreament(self, inputs, training):
        imgs, labs = inputs
        transform = self.train_transform if training else self.test_transform
        imgs = transform(ts2np(imgs))
        imgs = torch.Tensor(imgs).float()
        return imgs.to(self.device), labs.to(self.device)

    def forward(self, inputs):
        ips, labs = inputs

        imgs = ips
        del ips

        outs = self.Backbone(imgs)  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image': imgs
            },
            'inference_feat': {
                'embeddings': embed,
                'logits': logits,
                'labels': labs
            }
        }
        return retval

    def save_ckpt(self, iteration):
        mkdir(osp.join(self.save_path, "checkpoints/"))
        checkpoint = {
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'iteration': iteration}
        torch.save(checkpoint,
                    osp.join(self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(self.save_name, iteration)))

    def _load_ckpt(self, save_name):
        load_ckpt_strict = self.restore_ckpt_strict

        checkpoint = torch.load(save_name, map_location=self.device)
        model_state_dict = checkpoint['model']

        if not load_ckpt_strict:
            self.msg_mgr.log_info("-------- Restored Params List --------")
            self.msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
                set(self.state_dict().keys()))))

        self.load_state_dict(model_state_dict, strict=load_ckpt_strict)
        if self.training:
            if not self.optimizer_reset and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Optimizer from %s !!!" % save_name)
            if not self.scheduler_reset and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Scheduler from %s !!!" % save_name)
        self.msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)

    def resume_ckpt(self, restore_hint):
        save_name = self.save_name
        save_name = osp.join(
        self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
        self.iteration = restore_hint
        self._load_ckpt(save_name)

    def train_step(self, loss_sum) -> bool:
        """Conduct loss_sum.backward(), self.optimizer.step() and self.scheduler.step().

        Args:
            loss_sum:The loss of the current batch.
        Returns:
            bool: True if the training is finished, False otherwise.
        """

        self.optimizer.zero_grad()
        if loss_sum <= 1e-9:
            self.msg_mgr.log_info("Find the loss sum less than 1e-9 but the training process will continue!")
        else:
            loss_sum.backward()
            self.optimizer.step()

        self.iteration += 1
        self.scheduler.step()
        return True

    def inference(self, loader=None):
        """Inference all the test data.

        Args:
            rank: the rank of the current process.Transform
        Returns:
            Odict: contains the inference results.
        """
        loader = self.test_loader if loader is None else loader
        total_size = len(loader)
        pbar = tqdm(total=total_size, desc='Transforming')
        batch_size = loader.batch_sampler.batch_size
        rest_size = total_size
        info_dict = Odict()
        for inputs in loader:
            ipts = self.inputs_pretreament(inputs, training=False)
            retval = self.forward(ipts)
            inference_feat = retval['inference_feat']
            for k, v in inference_feat.items():
                inference_feat[k] = ts2np(v)
            info_dict.append(inference_feat)
            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v
        return info_dict

    @ staticmethod
    def run_train(model):
        """Accept the instance object(model) here, and then run the train loop."""
        for inputs in model.train_loader:
            ipts = model.inputs_pretreament(inputs, training=True)
            retval = model(ipts)
            training_feat, visual_summary = retval['training_feat'], retval['visual_summary']
            del retval
            loss_sum, loss_info = model.loss_aggregator(training_feat)
            ok = model.train_step(loss_sum)
            if not ok:
                continue

            visual_summary.update(loss_info)
            visual_summary['scalar/learning_rate'] = model.optimizer.param_groups[0]['lr']

            model.msg_mgr.train_step(loss_info, visual_summary)
            if model.iteration % model.save_iter == 0:
                # save the checkpoint
                model.save_ckpt(model.iteration)

                # run test if with_test = true
                if model.with_test:
                    model.msg_mgr.log_info("Running test...")
                    model.eval()
                    result_dict = Baseline.run_test(model)
                    model.train()
                    model.msg_mgr.reset_time()
            if model.iteration >= model.total_iter:
                break

    @ staticmethod
    def run_test(model):
        """Accept the instance object(model) here, and then run the test loop."""
        with torch.no_grad():
            info_dict = model.inference()

        logits = info_dict['logits']
        labels = info_dict['labels']

        pred = np.sum(logits, axis=2)
        pred = np.argmax(pred, axis=1)

        accu = (pred == labels).mean()
        model.msg_mgr.log_info(f'Accuracy: {accu*100}%')

        return info_dict
