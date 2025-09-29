import numpy as np
from functions import load_raw_univariate_ts
from models import analyze_classification, NoFussCrossEntropyLoss, Transformer_vanilla, ProbAttention,Transformer_performer,Transformer_darker,FourierCrossAttentionW
import torch
import logging
from tqdm import tqdm
import torch.utils.data as Data
import time
from torch.utils.data import Dataset, DataLoader
import argparse


logging.basicConfig(filename='new.log', filemode='w', format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="BinaryHeartbeat", #NATOPS
                    help='time series dataset. Options: See the datasets list')


parser.add_argument('--epochs', type=int, default="100", #NATOPS
                    help='time series dataset. Options: See the datasets list')

parser.add_argument('--batch_size', type=int, default="4", #NATOPS
                    help='time series dataset. Options: See the datasets list')

parser.add_argument('--number_att', type=int, default="1", #NATOPS
                    help='time series dataset. Options: See the datasets list')


parser.add_argument('--model_name', type=str, default="full", #NATOPS
                    help='time series dataset. Options: See the datasets list')

parser.add_argument('--learn_rate', type=float, default="0.001", #NATOPS
                    help='time series dataset. Options: See the datasets list')

parser.add_argument('--data_path', type=str, default="./data/raw/", #NATOPS
                    help='time series dataset. Options: See the datasets list')

parser.add_argument('--sketch_path', type=str, default="./data/tsketch/", #NATOPS
                    help='time series dataset. Options: See the datasets list')


args = parser.parse_args()


dataset = args.dataset
model_name = args.model_name


epochs = args.epochs
batch_size = args.batch_size

data_path = args.data_path
sketch_path = args.sketch_path

number_att = args.number_att
learn_rate = args.learn_rate


d_length = 4
d_model = 256



logging.basicConfig(filename='new.log', filemode='w', format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

window_size_rate = 3
train_data, test_data, train_label, test_label, dim_length, nclass = load_raw_univariate_ts(data_path, dataset)


train_shapes = np.load(sketch_path + dataset + '/X_train.npy')
test_shapes = np.load(sketch_path + dataset +'/X_test.npy')





number_shape = train_shapes.shape[1] # number of shapes per instance
d_max_len = train_shapes.shape[2] # length of shape after padding



m = 128

if model_name == 'darker':
    model = Transformer_darker(number_att, d_max_len, d_length, d_model, number_shape, nclass)
elif model_name == 'positive':
    model = Transformer_performer(number_att, d_max_len, d_length, d_model, number_shape, nclass, m)
elif model_name == 'full':
    model = Transformer_vanilla(number_att, d_max_len, d_length, d_model, number_shape, nclass)
elif model_name == 'informer':
    model = ProbAttention(number_att, d_max_len, d_length, d_model, number_shape, nclass)
elif model_name == 'fedformer':
    model = FourierCrossAttentionW(number_att, d_max_len, d_length, d_model, number_shape, nclass)




train_dataset = Data.TensorDataset(torch.FloatTensor(train_shapes), (torch.FloatTensor(train_label)).unsqueeze(1))

test_dataset = Data.TensorDataset(torch.FloatTensor(test_shapes), (torch.FloatTensor(test_label)).unsqueeze(1))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader =DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


# device
device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
#device = torch.device('cpu')
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)


loss_module = NoFussCrossEntropyLoss(reduction='none')



train_time = 0
for epoch in tqdm(range(1, epochs + 1), desc='Training Epoch', leave=False):
    epoch_start_time = time.time()
    model.train()
    total_samples = 0
    epoch_loss = 0
    for i, a in enumerate(train_loader):
        X, label = a
        label = label.to(device)
        prediction = model(X.to(device))
        loss = loss_module(prediction, label)
        batch_loss = torch.sum(loss)
        mean_loss = batch_loss / len(loss)
        optimizer.zero_grad()
        mean_loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        optimizer.step()

        total_samples += len(loss)
        epoch_loss += batch_loss.item()

    epoch_runtime = time.time() - epoch_start_time
    train_time = train_time + epoch_runtime
    epoch_loss = epoch_loss / total_samples
    logger.info("Epoch runtime: {} seconds\n".format(epoch_runtime))
    print('epoch : {}, train loss : {:.4f}' \
            .format(epoch, epoch_loss))
    if epoch == epochs or (epoch == 20) or (epoch == 50):
        logger.info("Evaluating on validation set ...")
        eval_start_time = time.time()
        model.eval()
        with torch.no_grad():
            epoch_loss = 0  # total loss of epoch
            total_samples = 0

            per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
            for i, a in enumerate(test_loader):
                X, label = a
                label = label.to(device)
                prediction = model(X.to(device))
                loss = loss_module(prediction, label)
                batch_loss = torch.sum(loss).cpu().item()
                mean_loss = batch_loss / len(loss)  # mean loss (over samples)

                per_batch['targets'].append(label.cpu().numpy())
                per_batch['predictions'].append(prediction.cpu().numpy())
                per_batch['metrics'].append([loss.cpu().numpy()])

            eval_time = time.time() - eval_start_time

            prediction = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
            probs = torch.nn.functional.softmax(prediction)
            prediction = torch.argmax(probs, dim=1).cpu().numpy()
            probs = probs.cpu().numpy()
            targets = np.concatenate(per_batch['targets'], axis=0).flatten()
            class_names = np.arange(probs.shape[1])
            accuracy = analyze_classification(prediction, targets, class_names)




