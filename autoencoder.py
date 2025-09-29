import numpy as np
import tslearn
from functions import get_window, sliding_all, load_raw_univariate_ts, generate_kernels_individual, sax, sax_to_tensor, stride_convolution, get_random_kernel
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import Counter
import torch.utils.data as Data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
#from model import Autoencoder
from model_mask import pad_and_create_mask
from model_2 import EncoderDecoder
from loss_functions import get_most_frequent_vector, get_triplet, Multi_level_TripletLoss
from sklearn.cluster import KMeans
import os
import pdb

seed = 42
np.random.seed(seed)


dataset_candidate_ucr = ['Rock','PigAirwayPressure','HandOutlines','BinaryHeartbeat','HouseTwenty','UrbanSound','CatsDogs']
data_path = './data_z_norm/'
dataset = 'AbnormalHeartbeat'
window_size_rate = 3


train_data, test_data, train_label, test_label, dim_length, nclass = load_raw_univariate_ts(data_path, dataset)



window_size = get_window(dim_length)

max_len = window_size[2]

step_size = [math.floor(window / window_size_rate) for window in window_size]


train_subsequences = sliding_all(train_data, window_size, step_size)




train_subsequences_combined_list = []
for array in train_subsequences:
    for subsequence in array:
        train_subsequences_combined_list.append(subsequence)

num_subsequences = len(train_subsequences_combined_list)


kernel_size =  3 # set default kernel size
kernel_num = 30  # set default kernel number
sigma = 10
confidence = 0.95
kernel_candidates = generate_kernels_individual(kernel_size, kernel_num, sigma, confidence)
kernel_set = kernel_candidates

kernel_set = torch.from_numpy((kernel_set[14:19]).astype(np.float32))

# set SAX parameters
segments = 4
alphabet_size = 4

sax_representations = []
for item in train_subsequences:
    sax_representations_per_window = [sax(ts, segments=segments, alphabet_size=alphabet_size) for ts in item]
    sax_representations.append(sax_representations_per_window)

a = 2

sax_counts = Counter(sax_representations[2])

max_sax_distance = segments * (alphabet_size - 1)

#
# plt.figure(figsize=(12, 6))
# plt.bar(sax_counts.keys(), sax_counts.values())
# plt.xlabel('SAX Representation')
# plt.ylabel('Frequency')
# plt.title('Frequency of SAX Representations')
# plt.xticks(rotation=90)
# plt.show()




# Convert SAX label to Tensor label (number)

sax_label_tensor_short = sax_to_tensor(sax_representations[0], segments, alphabet_size)
sax_label_tensor_medium = sax_to_tensor(sax_representations[1], segments, alphabet_size)
sax_label_tensor_long = sax_to_tensor(sax_representations[2], segments, alphabet_size)




# Set model parameter

input_channels = 1

output_channels = input_channels
num_res_blocks = 3
dilation_rate = 2


#model = Autoencoder(input_channels, output_channels, num_res_blocks, dilation_rate)

#model = EncoderDecoder(input_channels=1, hidden_channels=64,num_resblocks = 2, latent_dim=32, max_len=max_len)

model = EncoderDecoder(input_channels = 1, hidden_channels = 64, num_resblocks = 2,
                 latent_dim = 32, output_channels = 1, output_length = max_len)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# Set loss hyparameters:
alpha = 10    # triplet loss parameter
triplet_loss_function = Multi_level_TripletLoss(margin = alpha, max_sax_distance = max_sax_distance)
Lambada = 0.5


# Set training paremeter
batch_size = 64
num_epochs = 30
# Set device GPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# transform from numpy float 64 to torch float 32, and reshape from number * length to number * 1 * length, where 1 is channel

tensor_subsequences_short = torch.from_numpy(train_subsequences[0][:, np.newaxis, :]).float()
tensor_subsequences_medium = torch.from_numpy(train_subsequences[1][:, np.newaxis, :]).float()
tensor_subsequences_long = torch.from_numpy(train_subsequences[2][:, np.newaxis, :]).float()


# prepare tensor dataset and dataloader

train_dataset_short = TensorDataset(tensor_subsequences_short, sax_label_tensor_short)
train_dataset_medium = TensorDataset(tensor_subsequences_medium, sax_label_tensor_medium)
train_dataset_long = TensorDataset(tensor_subsequences_long, sax_label_tensor_long)


train_loader_short = DataLoader(dataset = train_dataset_short, batch_size=batch_size, shuffle=True)
train_loader_medium = DataLoader(dataset = train_dataset_medium, batch_size=batch_size, shuffle=True)
train_loader_long = DataLoader(dataset = train_dataset_long, batch_size=batch_size, shuffle=True)

test_loader_short = DataLoader(dataset = train_dataset_short, batch_size=1, shuffle=False)
test_loader_medium = DataLoader(dataset = train_dataset_medium, batch_size=1, shuffle=False)
test_loader_long = DataLoader(dataset = train_dataset_long, batch_size=1, shuffle=False)




for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    total_loss = 0

    model.train()
    for inputs, labels in train_loader_short:

        optimizer.zero_grad()
        inputs, mask = pad_and_create_mask(inputs, max_len)

        inputs = inputs.to(device)
        # to model
        encoded, decoded = model(inputs)

        # compute loss reconstruction based on 'decoded' and 'inputs'
        mask = ((mask[0,:,:]).to(device)).unsqueeze(0)  # to reduce memorary
        loss_reconstruction = criterion(decoded * mask,  inputs * mask)


        # [compute reconstruction loss]
        #loss_reconstruction = criterion(decoded, inputs)



        # choose anchor
        encoded = encoded.squeeze(1)

        anchor_label, anchor_indices = get_most_frequent_vector(labels)


        if len(anchor_indices) == 1:
            loss_triplet = 0

        # choose anchor, pos, neg, and get distance
        anchor, positives, negatives, negative_sax_distances = get_triplet(anchor_indices, encoded, labels, anchor_label)

        # [compute triplet loss]
        if isinstance(positives, torch.Tensor):
            loss_triplet = triplet_loss_function(anchor, positives, negatives, negative_sax_distances)
        else:     # if there is no other sample in same bucket with anchor
            loss_triplet = 0

        loss = Lambada * loss_triplet + (1 - Lambada) * loss_reconstruction
        loss.backward()
        optimizer.step()

        total_loss += loss.item()



        #print(f'Epoch [{epoch+1}/5], Iteration [{i+1}], Short, Loss: {loss.item():.4f}')

    for inputs, labels in train_loader_medium:

        optimizer.zero_grad()  # 清空梯度
        inputs, mask = pad_and_create_mask(inputs, max_len)

        inputs = inputs.to(device)

        # to model
        encoded, decoded = model(inputs)

        # compute loss reconstruction based on 'decoded' and 'inputs'
        mask = ((mask[0,:,:]).to(device)).unsqueeze(0)  # to reduce memorary
        loss_reconstruction = criterion(decoded * mask, inputs * mask)

        # [compute reconstruction loss]
        #loss_reconstruction = criterion(decoded, inputs)

        # choose anchor
        encoded = encoded.squeeze(1)

        anchor_label, anchor_indices = get_most_frequent_vector(labels)

        # if there is no other sample in same bucket with anchor
        if len(anchor_indices) == 1:
            loss_triplet = 0

        # choose anchor, pos, neg, and get distance
        anchor, positives, negatives, negative_sax_distances = get_triplet(anchor_indices, encoded, labels,
                                                                           anchor_label)



        ## Noted 1. random Conv1D, 2. kernel anchor to model 3.

        # 1. random Conv1D : stride = 2 because medium
        random_kernel = (get_random_kernel(kernel_set)).to(device)
        stride = 2
        a = inputs[anchor_indices, :, :]
        anchor_after_kernel = stride_convolution(inputs[anchor_indices[0], :, :], random_kernel, stride)
        anchor_after_kernel = anchor_after_kernel.unsqueeze(dim=0)
        # 2. kernel anchor to model
        anchor_after_kernel, _ = pad_and_create_mask(anchor_after_kernel, max_len) # pad anchor after kernel
        encoded_anchor_after_kernel, decoded_anchor_after_kernel = model(anchor_after_kernel)
        # 3. compute loss_kernel and loss_kernel_reconstruction
        loss_kernel = criterion(anchor, encoded_anchor_after_kernel.squeeze(dim=0))
        #loss_kernel_reconstruction =

        # [compute triplet loss]

        if isinstance(positives, torch.Tensor):
            loss_triplet = triplet_loss_function(anchor, positives, negatives, negative_sax_distances) + loss_kernel
        else:  # no subsequences in same bucket, let loss_triplet = loss_kernel
            loss_triplet = loss_kernel


        loss = Lambada * loss_triplet + (1 - Lambada) * loss_reconstruction
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        #print(f'Epoch [{epoch + 1}/5], Iteration [{i + 1}], Medium, Loss: {loss.item():.4f}')

    for inputs, labels in train_loader_long:

        optimizer.zero_grad()
        inputs, mask = pad_and_create_mask(inputs, max_len)

        inputs = inputs.to(device)

        # to model
        encoded, decoded = model(inputs)

        # compute loss reconstruction based on 'decoded' and 'inputs'
        mask = ((mask[0,:,:]).to(device)).unsqueeze(0)  # to reduce memorary
        loss_reconstruction = criterion(decoded * mask, inputs * mask)

        # [compute reconstruction loss]
        #loss_reconstruction = criterion(decoded, inputs)

        # choose anchor
        encoded = encoded.squeeze(1)

        anchor_label, anchor_indices = get_most_frequent_vector(labels)

        # if there is no other sample in same bucket with anchor
        if len(anchor_indices) == 1:
            loss_triplet = 0

        # choose anchor, pos, neg, and get distance
        anchor, positives, negatives, negative_sax_distances = get_triplet(anchor_indices, encoded, labels,
                                                                           anchor_label)

        ## Noted 1. random Conv1D, 2. kernel anchor to model 3.

        # 1. random Conv1D : stride = 2 because medium
        random_kernel = (get_random_kernel(kernel_set)).to(device)
        stride = 3
        a = inputs[anchor_indices, :, :]
        anchor_after_kernel = stride_convolution(inputs[anchor_indices[0], :, :], random_kernel, stride)
        anchor_after_kernel = anchor_after_kernel.unsqueeze(dim=0)
        # 2. kernel anchor to model
        anchor_after_kernel, _ = pad_and_create_mask(anchor_after_kernel, max_len)  # pad anchor after kernel
        encoded_anchor_after_kernel, decoded_anchor_after_kernel = model(anchor_after_kernel)
        # 3. compute loss_kernel and loss_kernel_reconstruction
        loss_kernel = criterion(anchor, encoded_anchor_after_kernel.squeeze(dim=0))

        #loss_kernel_reconstruction =

        # [compute triplet loss]

        if isinstance(positives, torch.Tensor):
            loss_triplet = triplet_loss_function(anchor, positives, negatives, negative_sax_distances) + loss_kernel
        else:  # no subsequences in same bucket, let loss_triplet = loss_kernel
            loss_triplet = loss_kernel


        loss = Lambada * loss_triplet + (1 - Lambada) * loss_reconstruction
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        #print(f'Epoch [{epoch + 1}/5], Iteration [{i + 1}], Long, Loss: {loss.item():.4f}')

    average_loss = total_loss / num_subsequences
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')


    #print('Training completed for epoch', epoch + 1)



# Evaluation

torch.save(model, 'model.pth')

#  model = torch.load('model.pth')

model.eval()
representations = []
with torch.no_grad():
    for inputs, labels in test_loader_short:
        inputs, mask = pad_and_create_mask(inputs, max_len)
        inputs = inputs.to(device)

        encoded, decoded = model(inputs)
        encoded = encoded.cpu()
        representations.append(encoded)
    for inputs, labels in test_loader_medium:
        inputs, mask = pad_and_create_mask(inputs, max_len)
        inputs = inputs.to(device)

        encoded, decoded = model(inputs)
        encoded = encoded.cpu()
        representations.append(encoded)
    for inputs, labels in test_loader_long:
        inputs, mask = pad_and_create_mask(inputs, max_len)
        inputs = inputs.to(device)

        encoded, decoded = model(inputs)
        encoded = encoded.cpu()
        representations.append(encoded)

# list to tensor
representations = torch.stack(representations)
# tensor to numpy
representations = representations.numpy()
representations = np.squeeze(representations, axis=1)

# set representations to different windows
split_indices = [train_subsequences[0].shape[0], train_subsequences[0].shape[0] + train_subsequences[1].shape[0], num_subsequences]
representations_all_window = np.split(representations, split_indices[:-1])

# set number of sketch
num_sketch = 1024
num_sketch_window = [int((3*num_sketch)/6), int((2*num_sketch)/6), int((1*num_sketch)/6)]


# model decoder to cpu
decoder = model.decoder
decoder = decoder.to('cpu')

a = 5

cluster_centers_all = []
centers_subsequences = []
for i in range (3):
    kmeans = KMeans(n_clusters=num_sketch_window[i], random_state=0)
    kmeans.fit(representations_all_window[i])


    cluster_centers = kmeans.cluster_centers_
    cluster_centers_all.append(cluster_centers)

    labels = kmeans.labels_     # obtain label for all subsequences
    cluster_index = 0
    indices = np.where(labels == cluster_index)[0]                # obtain subsequences label for a specific center

    cluster_centers_tensor = torch.from_numpy((cluster_centers).astype(np.float32))
    cluster_centers_tensor = cluster_centers_tensor.unsqueeze(1)
    model.decoder.eval()
    with torch.no_grad():
        output = decoder(cluster_centers_tensor)
        output = output.squeeze(1)
        output = output[:,:window_size[i]]
        output = output.numpy()
    centers_subsequences.append(output)



center_path = './data/data_center/' + dataset
os.makedirs(center_path, exist_ok=True)


np.savez(center_path + '/' + 'prototypes.npz', short=centers_subsequences[0], medium=centers_subsequences[1], long=centers_subsequences[2])















