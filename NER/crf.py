import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

START_TAG = -2
STOP_TAG = -1


class CRF(nn.Module):
    def __init__(self, target_size, gpu):
        super(CRF, self).__init__()
        self.target_size = target_size
        self.gpu = gpu
        transition_mat = torch.zeros(self.target_size + 2, self.target_size + 2)
        transition_mat[:, START_TAG] = -10000.0
        transition_mat[STOP_TAG, :] = -10000.0
        transition_mat[:, 0] = -10000.0  # pad index
        transition_mat[0, :] = -10000.0
        if self.gpu:
            transition_mat = transition_mat.cuda()
        self.transitions = nn.Parameter(transition_mat)

    def forward(self, input, mask, tags):
        forward_scores, score = self.cal_forward_score(input, mask)
        gold_score = self.cal_gold_score(score, mask, tags)
        loss = forward_scores - gold_score
        return loss

    def cal_forward_score(self, input, mask):
        batch_size = input.size(0)
        seq_len = input.size(1)
        tag_size = input.size(2)
        assert (tag_size == self.target_size+2)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        inputs = input.transpose(1, 0).contiguous().view(ins_num, 1,
                                                        tag_size).expand(ins_num, tag_size, tag_size)
        # emit score + T
        scores = inputs + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)
        # from start tag to word
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size, 1)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size,
                                                                                                  tag_size)
            cur_partition = log_sum_up(cur_values, tag_size)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
            partition.masked_scatter_(mask_idx, masked_cur_partition)

        # from last word to stop tag
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size,
                                                                         tag_size) + partition.contiguous().view(
            batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_up(cur_values, tag_size)
        final_partition = cur_partition[:, STOP_TAG]
        return final_partition.sum(), scores

    def cal_gold_score(self, scores, mask, tags):
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len))
        if self.gpu:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]
        end_transition = self.transitions[:, STOP_TAG].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        # cal end word id,length-1
        end_ids = torch.gather(tags, 1, length_mask - 1)
        end_energy = torch.gather(end_transition, 1, end_ids)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        return tg_energy.sum() + end_energy.sum()

    def viterbi_decode(self, feats, mask):
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert (tag_size == self.target_size + 2)
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        seq_iter = enumerate(scores)
        back_points = list()
        partition_history = list()
        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size)
        partition_history.append(partition)
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size,
                                                                                                tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1, 0).contiguous()
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, 1)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1, tag_size,
                                                                                                    tag_size).expand(
            batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        if self.gpu:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        pointer = last_bp[:, STOP_TAG]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1, 0).contiguous()
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        if self.gpu:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.detach().view(batch_size)
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx


def log_sum_up(vec, m_size):
    max_score, idx = torch.max(vec, 1)
    max_score_broadcast = max_score.unsqueeze(1).view(-1, 1, m_size).expand(-1, m_size, m_size)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 1))
