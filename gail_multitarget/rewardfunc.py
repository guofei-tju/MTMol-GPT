import torch
from torch.nn.utils.rnn import pad_sequence


class RewardFunction:
    def __init__(self, vocab, rewardnet, device):
        self.rewardnet = rewardnet
        self.vocab = vocab
        self.device = device

    def fit(self, demo1_smiles, demo2_smiles, model_smiles):
        self.rewardnet.train()
        # D_demo processing
        loss = self.get_reward_gru(demo1_smiles, demo2_smiles, model_smiles)['loss']
        self.rewardnet.optimizer.zero_grad()
        loss.backward()
        self.rewardnet.optimizer.step()

        return loss.item()

    def get_reward_gru(self, demo1_smiles, demo2_smiles, samples):
        demo1_ids = [torch.tensor(self.vocab.encode(string, add_bos=True, add_eos=True), dtype=torch.long,
                                  device=self.device) for string in demo1_smiles]
        demo1_pad_ids = pad_sequence(demo1_ids, batch_first=True,
                                     padding_value=self.vocab.pad)

        demo2_ids = [torch.tensor(self.vocab.encode(string, add_bos=True, add_eos=True), dtype=torch.long,
                                  device=self.device) for string in demo2_smiles]
        demo2_pad_ids = pad_sequence(demo2_ids, batch_first=True,
                                     padding_value=self.vocab.pad)

        samples_ids = [torch.tensor(self.vocab.encode(string, add_bos=True, add_eos=True), dtype=torch.long,
                                    device=self.device) for string in samples]
        samples_pad_ids = pad_sequence(samples_ids, batch_first=True,
                                       padding_value=self.vocab.pad)

        results = self.rewardnet.compute_reward_gru(demo1_pad_ids, demo2_pad_ids, samples_pad_ids)
        return results

    def get_reward(self, demo1_smiles, demo2_smiles, samples):
        return self.get_reward_gru(demo1_smiles, demo2_smiles, samples)['reward']
