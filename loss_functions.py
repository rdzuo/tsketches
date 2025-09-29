import torch
import torch.nn as nn
import torch.nn.functional as F


def get_most_frequent_vector(tensor):

    vectors = [tuple(row.tolist()) for row in tensor]


    vector_counts = {}
    for i, vector in enumerate(vectors):
        if vector in vector_counts:
            vector_counts[vector].append(i)
        else:
            vector_counts[vector] = [i]


    most_frequent_vector_tuple = max(vector_counts, key=lambda k: len(vector_counts[k]))
    indices = vector_counts[most_frequent_vector_tuple]


    most_frequent_vector_tensor = torch.tensor(most_frequent_vector_tuple, dtype=torch.int32)

    return most_frequent_vector_tensor, indices

def compute_label_distance(label_1, label_2):
    label_1 - label_2




def get_triplet(anchor_indices, inputs, labels, anchor_label):

    batch_size = inputs.shape[0]
    number_positive = len(anchor_indices) - 1
    number_negative = batch_size - len(anchor_indices)
    number_triplet = min(number_positive, number_negative)
    positives = []
    negatives = []
    negative_distances = []
    for i in range(batch_size):
        label = labels[i]
        if torch.equal(label, anchor_label):
            positive = inputs[i,:]
            positives.append(positive)
        else:
            negative = inputs[i,:]
            negative_distance = torch.norm(label.to(torch.float32) - anchor_label.to(torch.float32), p=1)
            negatives.append(negative)
            negative_distances.append(negative_distance)

    anchor = positives[0]
    positives = positives[1:number_triplet + 1]
    negatives = negatives[0:number_triplet]

    if len(positives) == 0:
        positives = 0
    else:
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)

        negative_distances = negative_distances[0:number_triplet]

    return anchor, positives, negatives, negative_distances


def collate_fn(batch, max_len):
    """动态填充批处理函数"""
    lengths = [x.shape[-1] for x in batch]
    # max_len = max(lengths)
    padded = torch.zeros(len(batch), 1, max_len)
    mask = torch.zeros(len(batch), 1, max_len)

    for i, x in enumerate(batch):
        padded[i, :, :lengths[i]] = x
        mask[i, :, :lengths[i]] = 1

    return mask, lengths, batch




class Multi_level_TripletLoss(nn.Module):
    def __init__(self, margin=1.0, max_sax_distance = 3.0 * 4.0):
        super(Multi_level_TripletLoss, self).__init__()
        self.margin = margin
        self.max_sax_distance = max_sax_distance

    def forward(self, anchor, positives, negatives, negative_sax_distances):

        number_triplet = positives.shape[0]
        margin = torch.tensor(self.margin, device=anchor.device)
        loss_all = torch.tensor(0, device= anchor.device)
        for i in range(number_triplet):
            positive = positives[i, :]
            negative = negatives[i, :]

            positive_distance = F.pairwise_distance(anchor, positive, p=2)
            negative_distance = F.pairwise_distance(anchor, negative, p=2)

            sax_distance = torch.tensor(negative_sax_distances[i], device=anchor.device)

            ratio = sax_distance / torch.tensor(self.max_sax_distance, device=anchor.device)

            loss_per = F.relu(positive_distance - negative_distance + ratio * margin)
            loss_all = loss_per + loss_all
        loss = loss_all / number_triplet


        return loss


