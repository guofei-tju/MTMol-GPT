import os
import sys

sys.path.append("..")
import pandas as pd
import numpy as np
import torch
import random
import wandb
from tqdm import tqdm, trange
import selfies as sf
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import logging
from replayBuffer import MolRLReplayBuffer
from rdkit.Chem import AllChem as Chem
from utils import canonical_smiles
from moses.metrics import FCDMetric

logger = logging.getLogger(__name__)


class GAILTrainerLoop:
    def __init__(self, config, vocab, model, reward_func, device='cpu', dataset1=None, dataset2=None,
                 valid_dataset1=None, valid_dataset2=None):
        self.config = config
        self.vocab = vocab
        self.model = model
        self.reward_func = reward_func
        self.n_gail = config.n_gail
        self.ppo_buffer_size = config.ppo_buffer_size
        self.ppo_mini_batch_size = config.ppo_mini_batch_size
        self.ppo_epsilon = config.ppo_epsilon
        self.ppo_iteration = config.ppo_iteration
        self.dis_nums = config.dis_nums
        self.mix_demo_ratio = config.mix_demo_ratio
        self.replay_buffer = MolRLReplayBuffer(self.ppo_buffer_size, shuffle=True)

        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.epoch_step = config.epoch_step
        self.warmup_step = config.warmup_step
        self.use_aug = config.use_augmentation
        self.use_sf = config.use_selfies

        self.org_dataset1 = dataset1
        self.org_dataset2 = dataset2

        self.dataset1_aug = pd.DataFrame()
        self.dataset2_aug = pd.DataFrame()

        self.valid_dataset1 = valid_dataset1
        self.valid_dataset2 = valid_dataset2

        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset1_train_loader = dataset1
        self.dataset2_train_loader = dataset2

        self.frac = 5
        self.device = device
        self.tmp_vars = {}

    def train(self, output_path, predictor1, predictor2):
        # Training begins
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for epoch in range(1, self.epochs + 1):
            logger.info(f'train epoch {epoch}')

            with torch.no_grad():
                self.model.generator.eval()

                if self.config.use_selfies:
                    smiles = self.model.generator.generate(100)
                    sequence = []
                    for sequence_i in smiles:
                        try:
                            sequence.append(sf.decoder(sequence_i))
                        except sf.DecoderError:
                            pass  # sf.encoder error!
                else:
                    sequence = self.model.generator.generate(100)
            new_smiles, valid_vec, valid_smiles = canonical_smiles(sequence)
            drd2_pre = predictor1(sequence)
            drd2_seq = [sequence[i] for i in list(np.array(np.where(drd2_pre >= 0.5)[0]))]
            htr1a_pre = predictor2(sequence)
            htr1a_seq = [sequence[i] for i in list(np.array(np.where(htr1a_pre >= 0.5)[0]))]
            if self.use_aug:
                if self.use_sf:
                    # convert smiles to selfies
                    drd2_sf = []
                    htr1a_sf = []
                    for drd2_se in drd2_seq:
                        try:
                            drd2_sf.append(sf.encoder(drd2_se))
                        except sf.DecoderError:
                            pass
                    for htr1a_se in htr1a_seq:
                        try:
                            htr1a_sf.append(sf.encoder(htr1a_se))
                        except sf.DecoderError:
                            pass
                    drd2_seq = drd2_sf
                    htr1a_seq = htr1a_sf

                self.dataset1_aug = pd.concat([self.dataset1_aug, pd.DataFrame(drd2_seq)])
                self.dataset1_aug = self.dataset1_aug.drop_duplicates()
                self.dataset1_aug = self.dataset1_aug[self.dataset1_aug.size - (
                    self.frac * self.org_dataset1.size if self.dataset1_aug.size > self.frac * self.org_dataset1.size else self.dataset1_aug.size):]
                self.dataset1 = pd.concat([self.org_dataset1, self.dataset1_aug], ignore_index=True)
                # self.dataset1 = pd.concat([self.dataset1, pd.DataFrame(drd2_seq)])
                self.dataset1 = self.dataset1.drop_duplicates()
                self.dataset1.to_csv(os.path.join(self.output_path, 'drd2_aug_datasets.csv'))

                self.dataset2_aug = pd.concat([self.dataset2_aug, pd.DataFrame(htr1a_seq)])
                self.dataset2_aug = self.dataset2_aug.drop_duplicates()
                self.dataset2_aug = self.dataset2_aug[self.dataset2_aug.size - (
                    self.frac * self.org_dataset2.size if self.dataset2_aug.size > self.frac * self.org_dataset2.size else self.dataset2_aug.size):]
                self.dataset2 = pd.concat([self.org_dataset2, self.dataset2_aug])
                # self.dataset2 = pd.concat([self.dataset2, pd.DataFrame(htr1a_seq)])
                self.dataset2 = self.dataset2.drop_duplicates()
                self.dataset2.to_csv(os.path.join(self.output_path, 'htr1a_aug_datasets.csv'))

            self.dataset1_train_loader = DataLoader(list(self.dataset1[0]), shuffle=True, pin_memory=True,
                                                    batch_size=self.batch_size, drop_last=True)

            mean_drd2 = np.mean(drd2_pre)
            mean_htr1a = np.mean(htr1a_pre)
            num_drd2 = np.sum(drd2_pre >= 0.5)
            num_htr1a = np.sum(htr1a_pre >= 0.5)
            both_sum = np.sum((htr1a_pre >= 0.5) & (drd2_pre >= 0.5))
            unique = len(np.unique(sequence))
            if self.config.use_wandb:
                # log the loss
                wandb.log(
                    {'test_vaild': len(valid_vec), 'test_drd2_mean_pre': mean_drd2, 'test_htr1a_mean_pre': mean_htr1a,
                     'test_drd2_num': num_drd2, 'drd2_dataset_num': len(self.dataset1),
                     'test_htr1a_num': num_htr1a, 'both_target_num': both_sum, 'test_unique': unique})
            if (epoch - 1) % 5 == 0:
                self.model.generator.save_model(
                    os.path.join(self.output_path, f'fine_tuning_generator_{epoch - 1}.pt'))
                self.model.discriminator.save_model(
                    os.path.join(self.output_path, f'fine_tuning_discriminator_{epoch - 1}.pt'))
            self.train_epoch(epoch)
            self.replay_buffer.clear()
            torch.cuda.empty_cache()

    def train_epoch(self, epoch):
        iter_train_dataloader1 = iter(self.dataset1_train_loader)
        for step in range(999):
            '''Buliding buffer'''
            buffer_count = 0
            while buffer_count < self.ppo_buffer_size:
                try:
                    batch1 = next(iter_train_dataloader1)
                except StopIteration:
                    return
                self.collect_samples(batch1)
                buffer_count += self.batch_size
            '''Rewardnet Training'''
            gail_batch_losses = []
            for i in range(self.n_gail):
                self.replay_buffer.restart()
                for mini_batch in self.replay_buffer.iterate_sample(self.ppo_mini_batch_size):
                    states1, actions, action_log_probs, rewards = zip(*mini_batch)
                    gail_loss = self.reward_func.fit(states1, actions)
                    gail_batch_losses.append(gail_loss)
                    if self.config.use_wandb:
                        # log the loss
                        wandb.log({'gail_loss': gail_loss})
                    logger.info(
                        f"rewardnet --mini-batch gail_loss:{gail_loss}")
                    torch.cuda.empty_cache()

            '''Generator Training'''
            for i in range(self.ppo_iteration):
                torch.cuda.empty_cache()
                mini_batch_gen_loss = []
                self.replay_buffer.restart()
                for mini_batch in self.replay_buffer.iterate_sample(self.ppo_mini_batch_size):
                    ppo_batch = {}
                    states1, actions, action_log_probs, rewards = zip(*mini_batch)
                    ppo_batch['generated_seq'] = {}
                    batch_idx = [torch.tensor(self.vocab.encode(string, add_bos=True, add_eos=True), dtype=torch.long,
                                              device=self.device) for string in actions]
                    ppo_batch['generated_seq']['index'] = pad_sequence([t[:-1] for t in batch_idx], batch_first=True,
                                                                       padding_value=self.vocab.pad)
                    ppo_batch['generated_seq']['label'] = pad_sequence([t[1:] for t in batch_idx], batch_first=True,
                                                                       padding_value=self.vocab.pad)
                    ppo_batch['generated_seq']['lens'] = torch.tensor([len(t) - 1 for t in batch_idx],
                                                                      dtype=torch.long, device=self.device)
                    ppo_batch["rewards"] = torch.FloatTensor(rewards).to(self.device)
                    ppo_batch["old_log_probs"] = torch.FloatTensor(action_log_probs).to(self.device)

                    gen_dict = self.train_generator_step(ppo_batch)
                    mini_batch_gen_loss.append(gen_dict["loss"])

                    if self.config.use_wandb:
                        wandb.log({'generator_loss': gen_dict["loss"]})
                    logger.info(
                        f'Generator --Epoch:{epoch}---loacl_step:{step}---loss:{gen_dict["loss"]}')

                generator_mean_loss = np.mean(mini_batch_gen_loss)
                if self.config.use_wandb:
                    wandb.log({'mini_batch_generator_mean_loss': generator_mean_loss,
                               'gail_batch_losses': np.mean(gail_batch_losses),
                               'lr': self.model.generator.optimizer.defaults['lr']})
                logger.info(
                    f"generator --mini-batch loss:{generator_mean_loss}")
            # self.mix_demo_ratio -= 0.001
            self.replay_buffer.clear()

    def train_generator_step(self, batch):
        self.model.generator.train()
        results = self.model.generator.compute_log_probs(batch['generated_seq'])
        # old log probilities
        log_probs = results["log_probs"]
        old_log_probs = batch["old_log_probs"]
        # # advantage
        advantages = batch["rewards"]
        # Policy Loss
        # shape: (batch)
        ratio = (log_probs - old_log_probs).exp()
        # ratio = log_probs
        ## shape: (batch)
        policy_loss1 = -advantages * ratio
        ## shape: (batch)
        policy_loss2 = -advantages * ratio.clamp(1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon)
        ## shape: (batch)
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()
        # loss = policy_loss
        loss = policy_loss

        # loss = -torch.mean(log_probs * advantages)
        # Backward Loss
        self.model.generator.optimizer.zero_grad()
        loss.backward()
        self.model.generator.optimizer.step()
        # self.model.generator.scheduler.step()
        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > self.ppo_epsilon).float().mean()
            approx_kl = (log_probs - old_log_probs).pow(2).mean()

        log_dict = {}
        log_dict["loss"] = loss.item()
        log_dict["policy_loss"] = policy_loss.item()
        log_dict["clip_frac"] = clip_frac.item()
        log_dict["approx_kl"] = approx_kl.item()
        log_dict["ratio"] = ratio.mean().item()
        log_dict["advantages"] = advantages.mean().item()
        if self.config.use_wandb:
            wandb.log(log_dict)
        return log_dict

    @torch.no_grad()
    def collect_samples(self, batch_target):
        num_one_demos = int(len(batch_target) * self.mix_demo_ratio)
        # Update Buffer for Demos sequences
        if num_one_demos > 0:
            batch_demo = batch_target[:num_one_demos]
            batch = {}
            batch_idx = [torch.tensor(self.vocab.encode(string, add_bos=True, add_eos=True), dtype=torch.long,
                                      device=self.device) for string in batch_demo]
            batch['index'] = pad_sequence([t[:-1] for t in batch_idx], batch_first=True,
                                          padding_value=self.vocab.pad)
            batch['label'] = pad_sequence([t[1:] for t in batch_idx], batch_first=True,
                                          padding_value=self.vocab.pad)
            batch['lens'] = torch.tensor([len(t) - 1 for t in batch_idx],
                                         dtype=torch.long, device=self.device)
            batch['prop'] = []
            log_probs = self.model.generator.compute_log_probs(batch)["log_probs"]
            demos_log_probs = log_probs.tolist()
            # rewards = self.reward_func.get_reward(batch_demo1, batch_demo2, batch_demo1).tolist()
            # rewards = np.array(rewards)
            rewards = np.ones((len(batch_demo))) * 2.0
            self.replay_buffer.update_batch(
                states=batch_demo,
                actions=batch_demo,
                action_log_probs=demos_log_probs,
                rewards=rewards
            )
        else:
            num_one_demos = 0
        # Update Buffer for Generations
        self.model.discriminator.eval()
        actual_sample_size = len(batch_target) - num_one_demos
        select_batch_target = batch_target[num_one_demos:num_one_demos + actual_sample_size]
        # actual_sample_size = len(batch_target1)
        # seq_ids, sequences, seqs_log_p, prop_matrix = self.model.generator.generation(actual_sample_size,
        #                                                                             self.prop)
        sequences = []
        seqs_log_p_sum = torch.Tensor().to(self.device)
        while len(sequences) < actual_sample_size:
            torch.cuda.empty_cache()
            sequence, seq_log_p, seq_log_p_sum = self.model.generator.generate_prob(actual_sample_size * 2)
            seq_log_p_sum = seq_log_p_sum.view(-1, 1)
            index = -1
            while len(sequences) < actual_sample_size:
                index += 1
                if index >= len(sequence):
                    break
                sm = sequence[index]
                try:
                    sm2 = sf.decoder(sm)
                    mol = Chem.MolFromSmiles(sm2)
                except:
                    continue
                if sm in sequences:
                    continue
                sequences.append(sm)
                seqs_log_p_sum = torch.cat([seqs_log_p_sum, seq_log_p_sum[index]])
        rewards = self.reward_func.get_reward(select_batch_target, sequences).tolist()
        rewards = np.array(rewards)
        self.replay_buffer.update_batch(
            states=select_batch_target,
            actions=sequences,
            action_log_probs=seqs_log_p_sum,
            rewards=rewards
        )
