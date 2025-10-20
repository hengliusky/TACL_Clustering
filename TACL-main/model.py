import fontTools.subset
from layers import *
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch


class FeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(1)) # 初始化权重

        self.softmax = nn.Softmax(dim=0)

    def forward(self, feat, feat_a, feat_b):
        weights = self.weights
        weights = torch.sigmoid(weights)
        fused_feat_a = weights * feat + (1 - weights)  * feat_a
        fused_feat_b = weights * feat + (1 - weights)  * feat_b
        print(weights, "weights")
        return fused_feat_a, fused_feat_b

def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def min_max_normalize(tensor, min_val=0.0, max_val=1.0):
    """
    将输入张量进行最小-最大归一化，将值缩放到[min_val, max_val]范围内。

    参数:
    tensor (torch.Tensor): 输入张量
    min_val (float): 归一化后的最小值
    max_val (float): 归一化后的最大值

    返回:
    torch.Tensor: 归一化后的张量
    """
    tensor_min = tensor.min(dim=0, keepdim=True)[0]
    tensor_max = tensor.max(dim=0, keepdim=True)[0]
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor * (max_val - min_val) + min_val


class Encoder_Net(nn.Module):
    def __init__(self, dims, cluster_num):
        super(Encoder_Net, self).__init__()
        self.layers1 = nn.Linear(dims[0], dims[1])
        self.low = nn.Linear(dims[1], cluster_num)
        self.initialize_weights()

    def forward(self, x):
        out1 = self.layers1(x)
        out1 = F.normalize(out1, dim=1, p=2)
        out1 = F.relu(out1)
        logits = self.low(out1)
        return out1, logits

    def initialize_weights(self):
        self.apply(he_init)  # 使用He初始化


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, cluster_num):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.gc = GraphConvolution(nfeat, out)
        self.low = nn.Linear(out, cluster_num)

    def forward(self, x, adj):
        x = self.gc(x, adj)
        logits = self.low(x)
        logits = F.normalize(logits, dim=1, p=2)
        return x, logits

    def get_emb(self, x, adj):
        return F.relu(self.gc1(x, adj)).detach()


def loss_cal(x, x_aug):
    T = 1.0
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss


class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, hidden_dims, output_dim, rank, use_softmax=False):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of audio, video and text

        self.feature1_hidden = hidden_dims
        self.feature2_hidden = hidden_dims

        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        self.feature1_factor = Parameter(torch.Tensor(self.rank, self.feature1_hidden + 1, self.output_dim))
        self.feature2_factor = Parameter(torch.Tensor(self.rank, self.feature2_hidden + 1, self.output_dim))

        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        xavier_normal(self.feature1_factor)
        xavier_normal(self.feature2_factor)
        xavier_normal(self.fusion_weights)
        print("feature1_factor", self.feature1_factor.shape)
        print("feature2_factor", self.feature2_factor.shape)
        self.fusion_bias.data.fill_(0)

    def forward(self, feature1, feature2):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        feature1_h = feature1
        feature2_h = feature2
        batch_size = feature1_h.data.shape[0]

        if feature1_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor


        _featue1_h = torch.cat((torch.ones(batch_size, 1).type(DTYPE), feature1_h), dim=1)
        _featue2_h = torch.cat((torch.ones(batch_size, 1).type(DTYPE), feature2_h), dim=1)

        fusion_feature1 = torch.matmul(_featue1_h, self.feature1_factor)
        fusion_feature2 = torch.matmul(_featue2_h, self.feature2_factor)
        fusion_zy = fusion_feature1 * fusion_feature2

        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        return output
