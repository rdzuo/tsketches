import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_window(l):
    if(l > 200):
        window_size = [50, 100, 200]
    elif (l > 10000):
        window_size = [300, 600, 900]

    return window_size

def load_raw_univariate_ts(path, dataset):
    path = path + dataset + "/"
    x_train = np.load(path + 'X_train.npy')
    y_train = np.load(path + 'y_train.npy')
    x_test = np.load(path + 'X_test.npy')
    y_test = np.load(path + 'y_test.npy')

    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)

    nclass = int(np.amax(y_train)) + 1
    dim_length = x_train.shape[1]

    return x_train, x_test, y_train, y_test, dim_length, nclass



def sliding_window(data, window_size, step_size):
    num_sequences = data.shape[0]
    sequence_length = data.shape[1]


    num_subsequences = (sequence_length - window_size) // step_size + 1


    subsequences = []

    for i in range(num_sequences):
        for j in range(0, num_subsequences * step_size, step_size):
            subsequence = data[i, j:j + window_size]
            subsequences.append(subsequence)

    return np.array(subsequences)



def sliding_all(data, window_size, step_size):
    subesequences_all = []
    for i in range (0,3):
        subsequences = sliding_window(data, window_size[i], step_size[i])
        subesequences_all.append(subsequences)

    return subesequences_all


def paa(ts, segments):
    """应用PAA进行降维"""
    n = len(ts)
    segment_size = n // segments
    paa_result = np.zeros(segments)
    for i in range(segments):
        start = i * segment_size
        end = start + segment_size
        paa_result[i] = np.mean(ts[start:end])
    return paa_result


def sax(ts, segments, alphabet_size):

    #ts = z_normalize(ts)


    paa_representation = paa(ts, segments)


    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])


    sax_representation = []
    for value in paa_representation:
        symbol = np.sum(breakpoints < value)
        sax_representation.append(chr(97 + symbol))  # 使用字母 'a', 'b', 'c', ...

    return ''.join(sax_representation)


def generate_kernels_individual(kernel_size, kernel_num, sigma, confidence):

    mean = 1/kernel_size

    stand_var = sigma
    # get z value from confidence
    z_value = norm.pdf(1-(1-confidence)/2)

    upper_bound = mean + z_value * stand_var
    lower_bound = mean - z_value * stand_var

    samples = []
    while len(samples) < kernel_num - 1:
        sample = np.random.normal(mean, sigma)
        if lower_bound <= sample <= upper_bound:
            samples.append(sample)
    # 确保中心点被选中
    samples.append(mean)
    samples = np.sort(samples)
    kernels = np.empty([kernel_num,kernel_size])
    for i in range(kernel_num):
        kernels[i,:] = samples[i]

    return kernels


def stride_convolution(x: torch.Tensor,
                       kernel: torch.Tensor,
                       stride: int,
                       bias: torch.Tensor = None) -> torch.Tensor:


    x = x.squeeze(0)
    assert len(x.shape) == 1, "must be 1"
    kernel_size = kernel.shape[0]
    if kernel_size < stride:
        raise ValueError("kernel_size >= stride")


    pad_total = kernel_size - stride
    left_pad = pad_total // 2
    right_pad = pad_total - left_pad


    device = x.device
    x_padded = F.pad(
        x.view(1, 1, -1),
        (left_pad, right_pad)
    )


    conv = nn.Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        bias=bias is not None
    ).to(device)


    conv.weight.data = kernel.view(1, 1, -1).to(device)
    if bias is not None:
        conv.bias.data = bias.view(-1).to(device)


    output = conv(x_padded)
    return output.squeeze(0)


# function for SAX to Tensor label
def sax_to_tensor(strings, segments, alphabet_size):

    alphabet = 'abcdefghijklmnopqrstuvwxyz'[:alphabet_size]
    char_to_index = {char: idx for idx, char in enumerate(alphabet)}


    tensors = torch.zeros((len(strings), segments), dtype=torch.int32)

    for i, string in enumerate(strings):
        for j, char in enumerate(string):
            if char in char_to_index:
                tensors[i, j] = char_to_index[char]
            else:
                raise ValueError(f"Character '{char}' is not in the first {alphabet_size} letters of the alphabet.")

    return tensors


def get_random_kernel(kernel_set):

    random_index = torch.randint(0, kernel_set.size(0), (1,)).item()

    return kernel_set[random_index]