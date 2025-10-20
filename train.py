import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import *
from tqdm import tqdm
from torch import optim
from model import *
import torch.nn.functional as F
import torch.nn as nn
# from layers import *
from sklearn.decomposition import PCA
import pickle
import warnings
from visualization import t_sne
from sklearn.metrics.pairwise import cosine_similarity

# parameter settings
parser = argparse.ArgumentParser()
parser.add_argument('--gnnlayers', type=int, default=5, help="Number of gnn layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--epochs2', type=int, default=200, help='Number of epochs to train2.')
parser.add_argument('--dims', type=int, default=800, help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=1, help='Loss balance parameter')
parser.add_argument('--beta', type=float, default=0.5, help='Loss balance parameter')
parser.add_argument('--threshold', type=float, default=0.80, help='the threshold')
parser.add_argument('--gama', type=float, default=0.5, help='Coefficient of weight')
parser.add_argument('--theta', type=float, default=1, help='Loss balance parameter')
parser.add_argument('--rm_pct', type=int, default=2, help='the threshold of removing')
parser.add_argument('--add_pct', type=int, default=57, help='the threshold of adding')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--cluster_num', type=int, default=7, help='number of clusters.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--device', type=str, default='cuda', help='the training device')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
warnings.filterwarnings('ignore', category=UserWarning)

# parameter settings
if args.dataset == 'cora':
    args.gnnlayers = 5 
    args.dims = 750 
    args.lr = 1e-4 
    args.cluster_num = 7
    args.add_pct = 57 
    args.rm_pct = 2 
    args.beta = 0.5
    args.theta = 1
    args.gama = 0.5

elif args.dataset == 'acm':
    args.gnnlayers = 6 
    args.dims = 500 
    args.epochs = 400 
    args.epochs2 = 200 
    args.lr = 1e-5  
    args.cluster_num = 3
    args.gama = 0.6 

    args.add_pct = 57  
    args.rm_pct = 5  
    # args.alpha = 0.5
    # args.beta = 1
    # args.theta = 0.5


elif args.dataset == 'amap':
    args.gnnlayers = 10
    args.dims = 400  
    args.lr = 1e-4
    args.cluster_num = 8
    args.add_pct = 57  
    args.rm_pct = 2 
    args.alpha = 1
    args.gama = 0.1 

    args.beta = 0.5
    args.theta = 1


elif args.dataset == 'bat':
    args.gnnlayers = 20  
    args.epochs = 300 
    args.epochs2 = 200  
    args.dims = 50
    args.lr = 1e-4 
    args.cluster_num = 4
    args.add_pct = 57  
    args.rm_pct = 1 
    args.gama = 0.2 

elif args.dataset == 'eat':
    args.gnnlayers = 20 
    args.epochs = 400 
    args.epochs2 = 200 
    args.dims = 100
    args.lr = 1e-4 
    args.cluster_num = 4
    args.add_pct = 47 
    args.rm_pct = 2 
    args.alpha = 1 
    args.beta = 0.5 
    args.theta = 1 
    args.gama = 0.5 

elif args.dataset == 'uat':
    args.gnnlayers = 24 
    args.dims = 100 
    args.epochs = 400
    args.epochs2 = 200 
    args.lr = 1e-4 
    args.cluster_num = 4
    args.add_pct = 57 
    args.rm_pct = 2 
    args.alpha = 1
    args.beta = 0.5
    args.theta = 1 
    args.gama = 0.2 


for seed in range(10):

    setup_seed(seed)

    features_o, true_labels, adj = load_graph_data(args.dataset)
    if features_o.dtype.kind in {'U', 'S'}: 
        features_o = features_o.astype(float)

    pca = PCA(n_components=args.dims)  # 降维
    features = pca.fit_transform(features_o)
    features = torch.FloatTensor(features)
    features_o = torch.FloatTensor(features_o)

    A_pred = pickle.load(
        open(f'data/' + args.dataset + '/edge_probabilities/' + args.dataset + '_graph_logits.pkl', 'rb'))
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj = sp.csr_matrix(adj)
    adj.eliminate_zeros()
    adj_tensor = torch.tensor(adj.todense(), dtype=torch.float32)

    adj_rm = sample_graph_det(adj, A_pred, args.rm_pct, 0)
    adj_add = sample_graph_det(adj, A_pred, 0, args.add_pct)

    adj_add = adj_add - sp.dia_matrix((adj_add.diagonal()[np.newaxis, :], [0]), shape=adj_add.shape)
    adj_add = sp.csr_matrix(adj_add)
    adj_add.eliminate_zeros()

    adj_rm = adj_rm - sp.dia_matrix((adj_rm.diagonal()[np.newaxis, :], [0]), shape=adj_rm.shape)
    adj_rm = sp.csr_matrix(adj_rm)
    adj_rm.eliminate_zeros()

    adj_add_norm = my_preprocess_graph(adj_add, norm='sym', renorm=True)
    adj_add_tensor = torch.tensor(adj_add_norm.todense(), dtype=torch.float32).cuda()
    adj_rm_norm = my_preprocess_graph(adj_rm, norm='sym', renorm=True)
    adj_rm_tensor = torch.tensor(adj_rm_norm.todense(), dtype=torch.float32).cuda()

    adj_norm = my_preprocess_graph(adj, norm='sym', renorm=True)
    sm_fea_s = sp.csr_matrix(features).toarray()
    

    for i in range(args.gnnlayers):
        sm_fea_s = adj_norm.dot(sm_fea_s)

    sm_fea_s = torch.FloatTensor(sm_fea_s)

    gcn_encoder = GCN(features.shape[1], features.shape[1], features.shape[1], args.cluster_num)
    if args.dataset == "eat" or args.dataset == "uat" or args.dataset == "acm":
        gcn_encoder = GCN(features_o.shape[1], features_o.shape[1], features.shape[1], args.cluster_num)

    optimizer_gcn_encoder = optim.Adam(gcn_encoder.parameters(), lr=args.lr, weight_decay=1e-5)

    model = Encoder_Net([features.shape[1]] + [features.shape[1]], args.cluster_num)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    linear = nn.Linear(args.dims, args.cluster_num)
    linear.cuda()
    linear.train()

    optimizer_linear = optim.Adam(linear.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer_linear.zero_grad()

    # GPU
    if args.cuda:
        gcn_encoder.cuda()
        model.cuda()
        # model_lmf.cuda()
        sm_fea_s = sm_fea_s.cuda()
        features_o = features_o.cuda()
        features = features.cuda()
        adj_rm_tensor = adj_rm_tensor.cuda()
        adj_add_tensor = adj_add_tensor.cuda()

    print('Start Training...')

    best_acc = 0
    all_gradients = []
    for epoch in tqdm(range(args.epochs)):

        model.train()
        gcn_encoder.train()
        # model_lmf.train()

        optimizer.zero_grad()
        optimizer_gcn_encoder.zero_grad()

        # obtain embedding
        z, logits_z = model(sm_fea_s)

        if args.dataset == "cora" or args.dataset == "bat" or args.dataset == "amap" or args.dataset == "dblp" or args.dataset == 'wiki':
            h1, _ = gcn_encoder(features, adj_add_tensor)
            h2, _ = gcn_encoder(features, adj_rm_tensor)
        elif args.dataset == "eat" or args.dataset == "uat" or args.dataset == "acm":
            h1, _ = gcn_encoder(features_o, adj_add_tensor)
            h2, _ = gcn_encoder(features_o, adj_rm_tensor)

        # contrastive
        contra_loss = loss_cal(z, h1) + loss_cal(z, h2)

        if epoch > args.epochs2:
            z1 = z * (1 - args.gama) + args.gama * h1
            z2 = z * (1 - args.gama) + args.gama * h2

            logits_z1 = linear(z1)
            logits_z1 = F.normalize(logits_z1, dim=1, p=2)
            logits_z2 = linear(z2)
            logits_z2 = F.normalize(logits_z2, dim=1, p=2)
            pseudo_z1 = torch.softmax(logits_z1, dim=-1)
            pseudo_z2 = torch.softmax(logits_z2, dim=-1)

            z_fuse_cluster = (z1 + z2) / 2

            _, _, _, _, predict_labels, centers, dis = clustering(z_fuse_cluster, true_labels, args.cluster_num)
            high_confidence = torch.min(dis, dim=1).values.cpu()
            threshold = torch.sort(high_confidence).values[int(len(high_confidence) * args.threshold)]
            high_confidence_idx = np.argwhere(high_confidence < threshold)[0]
            h_i = high_confidence_idx.numpy()
            y_sam = torch.tensor(predict_labels, device=args.device)[high_confidence_idx]

            loss_match = (F.cross_entropy(pseudo_z1[h_i], y_sam)).mean() + (
                F.cross_entropy(pseudo_z2[h_i], y_sam)).mean()

            kl_loss = pq_loss_func(z_fuse_cluster, centers)

            # total loss
            total_loss = contra_loss + args.beta * loss_match + args.theta * kl_loss

        else:
            total_loss = contra_loss

        total_loss.backward()

        optimizer.step()
        optimizer_gcn_encoder.step()
        optimizer_linear.step()

        if epoch % 1 == 0:
            model.eval()
            linear.eval()

            z1 = z * (1 - args.gama) + args.gama * h1
            z2 = z * (1 - args.gama) + args.gama * h2
            hidden_emb = (z1 + z2) / 2

            acc, nmi, ari, f1, predict_labels, centers, dis = clustering(hidden_emb, true_labels, args.cluster_num)

            if acc >= best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1
                best_embed = hidden_emb

        if epoch == args.epochs - 1:
            t_sne(best_embed.detach().cpu().numpy(), true_labels, features.shape[0], True, str(seed))

    acc_list.append(best_acc)
    nmi_list.append(best_nmi)
    ari_list.append(best_ari)
    f1_list.append(best_f1)

    tqdm.write("Optimization Finished!")
    tqdm.write('best_acc: {}, best_nmi: {}, best_ari: {}, best_f1: {}'.format(best_acc, best_nmi, best_ari, best_f1))

acc_list = np.array(acc_list)
nmi_list = np.array(nmi_list)
ari_list = np.array(ari_list)
f1_list = np.array(f1_list)

print("ACC ", acc_list.mean(), "±", acc_list.std())
print("NMI ", nmi_list.mean(), "±", nmi_list.std())
print("ARI ", ari_list.mean(), "±", ari_list.std())
print("F1 ", f1_list.mean(), "±", f1_list.std())
