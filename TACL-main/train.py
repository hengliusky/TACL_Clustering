import argparse
import time
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
# parser.add_argument('--linlayers', type=int, default=1, help="Number of hidden layers")
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--epochs2', type=int, default=200, help='Number of epochs to train2.')
# parser.add_argument('--hidden', type=int, default=128, help='hidden_num')
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
    args.gnnlayers = 5  # 5
    args.dims = 750  # 750
    args.lr = 1e-4  # 1e-4
    args.cluster_num = 7
    args.add_pct = 57  # 57
    args.rm_pct = 2  # 2

    args.beta = 0.5
    args.theta = 1
    args.gama = 0.5

elif args.dataset == 'acm':
    args.gnnlayers = 6 #6
    args.dims = 500 # 500
    args.epochs = 400 # 500
    args.epochs2 = 200 # 200
    args.lr = 1e-5  # 1e-5
    args.cluster_num = 3
    args.gama = 0.6 # 0.6

    args.add_pct = 57  # 57
    args.rm_pct = 5  # 5
    # args.alpha = 0.5
    # args.beta = 1
    # args.theta = 0.5


elif args.dataset == 'amap':
    args.gnnlayers = 10
    args.dims = 400  # 450
    args.lr = 1e-4
    args.cluster_num = 8
    args.add_pct = 57  # 47
    args.rm_pct = 2 # 2
    args.alpha = 1
    args.gama = 0.1 # 0.1

    args.beta = 0.5
    args.theta = 1


elif args.dataset == 'bat':
    args.gnnlayers = 20  # 20
    args.epochs = 300  # 300
    args.epochs2 = 200  #250
    args.dims = 50
    args.lr = 1e-4  # 1e-4
    args.cluster_num = 4
    args.add_pct = 57  # 40
    args.rm_pct = 1  # 1
    args.gama = 0.2 # 0.2

elif args.dataset == 'eat':
    args.gnnlayers = 20  # 10 20
    args.epochs = 400  # 400
    args.epochs2 = 200  # 200
    args.dims = 100
    args.lr = 1e-4  # 1e-3
    args.cluster_num = 4
    args.add_pct = 47  # 39
    args.rm_pct = 2 # 1
    args.alpha = 1  # 0.5
    args.beta = 0.5  # 无变化
    args.theta = 1  # 无变化
    args.gama = 0.5  # 0.5

elif args.dataset == 'uat':
    args.gnnlayers = 24  # 24
    args.dims = 100 # 100
    args.epochs = 400 # 500
    args.epochs2 = 200 # 250
    args.lr = 1e-4  # 1e-3
    args.cluster_num = 4
    args.add_pct = 57  # 35
    args.rm_pct = 2  # 1
    args.alpha = 1
    args.beta = 0.5
    args.theta = 1 # 0.5
    args.gama = 0.2 # 0.2



acc_list = []
nmi_list = []
ari_list = []
f1_list = []
dt_list = []

# acc_list_z1 = []
# nmi_list_z1 = []
# ari_list_z1 = []
# f1_list_z1 = []
#
# acc_list_z2 = []
# nmi_list_z2 = []
# ari_list_z2 = []
# f1_list_z2 = []
#
# acc_list_z = []
# nmi_list_z = []
# ari_list_z = []
# f1_list_z = []

for seed in range(10):

    setup_seed(seed)

    # adj, features, true_labels, idx_train, idx_val, idx_test = load_data(args.dataset)
    features_o, true_labels, adj = load_graph_data(args.dataset)
    # 检查数组是否包含字符串并将其转换为浮点数
    if features_o.dtype.kind in {'U', 'S'}:  # 'U' 和 'S' 表示 Unicode 和字节字符串
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
    print("原始图大小" + str(len(adj.data)))
    adj_tensor = torch.tensor(adj.todense(), dtype=torch.float32)

    # 创建增删视图
    adj_rm = sample_graph_det(adj, A_pred, args.rm_pct, 0)
    adj_add = sample_graph_det(adj, A_pred, 0, args.add_pct)

    # adj_rm, adj_add = sample_graph_threshold(adj, A_pred, 0.5, 0.9)

    print("add图大小：" + str(len(adj_add.data)), "  rm图大小:" + str(len(adj_rm.data)))
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


    # # 确保矩阵对称
    # adj_rm_symmetric = adj_rm.maximum(adj_rm.T)
    # adj_add_symmetric = adj_rm.maximum(adj_add.T)
    #
    # # 将矩阵转换为原始邻接矩阵的格式（csr_matrix 格式）
    # adj_rm_csr = sp.csr_matrix(adj_rm_symmetric)
    # adj_add_csr = sp.csr_matrix(adj_add_symmetric)
    #
    # # 将邻接矩阵转换为 numpy 数组并保存为 .npy 文件
    # np.save('adj_rm_'+args.dataset+'.npy', adj_rm_csr.toarray())
    # np.save('adj_add_'+args.dataset+'.npy', adj_add_csr.toarray())
    

    for i in range(args.gnnlayers):
        sm_fea_s = adj_norm.dot(sm_fea_s)

    sm_fea_s = torch.FloatTensor(sm_fea_s)

    gcn_encoder = GCN(features.shape[1], features.shape[1], features.shape[1], args.cluster_num)
    if args.dataset == "eat" or args.dataset == "uat" or args.dataset == "acm":
        gcn_encoder = GCN(features_o.shape[1], features_o.shape[1], features.shape[1], args.cluster_num)

    optimizer_gcn_encoder = optim.Adam(gcn_encoder.parameters(), lr=args.lr, weight_decay=1e-5)

    model = Encoder_Net([features.shape[1]] + [features.shape[1]], args.cluster_num)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # model_lmf = LMF(features.shape[1], 256, 256)
    # factors = list(model_lmf.parameters())[:2]
    # other = list(model_lmf.parameters())[2:]
    # optimizer_lmf = optim.Adam([{"params": factors, "lr": 0.01}, {"params": other, "lr": 0.01}],
    #                        weight_decay=0.01)  # don't optimize the first 2 params, they should be fixed (output_range and shift)

    linear = nn.Linear(args.dims, args.cluster_num)
    linear.cuda()
    linear.train()

    # fuse_linear = nn.Linear(args.dims * 2, args.dims)
    # fuse_linear.cuda()
    # fuse_linear.train()

    optimizer_linear = optim.Adam(linear.parameters(), lr=args.lr, weight_decay=1e-5)
    # optimizer_fuse_linear = optim.Adam(fuse_linear.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer_linear.zero_grad()
    # optimizer_fuse_linear.zero_grad()


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
    best_acc_z1 = 0
    best_acc_z2 = 0
    best_acc_z = 0
    best_embed = 0

    start = time.time()
    all_gradients = []
    for epoch in tqdm(range(args.epochs)):

        model.train()
        gcn_encoder.train()
        # model_lmf.train()

        optimizer.zero_grad()
        optimizer_gcn_encoder.zero_grad()
        # optimizer_lmf.zero_grad()

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

        # if epoch == args.epochs2:
        #     cos_12 = cosine_similarity(h1.detach().cpu(), z.detach().cpu())
        #     cos_21 = cosine_similarity(h2.detach().cpu(), z.detach().cpu())
        #
        #     sim1 = np.diag(cos_12)
        #     sim2 = np.diag(cos_21)
        #     mean_sim1 = np.mean(sim1)
        #     mean_sim2 = np.mean(sim2)
        #
        #     print("mean_sim1", "mean_sim2", mean_sim1, mean_sim2)

        if epoch > args.epochs2:
            z1 = z * (1 - args.gama) + args.gama * h1
            z2 = z * (1 - args.gama) + args.gama * h2
            # z2 = z + args.gama * h2


            logits_z1 = linear(z1)
            logits_z1 = F.normalize(logits_z1, dim=1, p=2)
            logits_z2 = linear(z2)
            logits_z2 = F.normalize(logits_z2, dim=1, p=2)
            pseudo_z1 = torch.softmax(logits_z1, dim=-1)
            pseudo_z2 = torch.softmax(logits_z2, dim=-1)

            z_fuse_cluster = (z1 + z2) / 2

            # z_fuse_cluster = model_lmf(z1, z2)

            # cat = torch.cat((z1, z2), dim=1)
            # z_fuse_cluster = fuse_linear(cat)

            _, _, _, _, predict_labels, centers, dis = clustering(z_fuse_cluster, true_labels, args.cluster_num)
            # _, _, _, _, predict_labels, centers, dis = clustering(z, true_labels, args.cluster_num)
            high_confidence = torch.min(dis, dim=1).values.cpu()
            threshold = torch.sort(high_confidence).values[int(len(high_confidence) * args.threshold)]
            high_confidence_idx = np.argwhere(high_confidence < threshold)[0]
            h_i = high_confidence_idx.numpy()
            y_sam = torch.tensor(predict_labels, device=args.device)[high_confidence_idx]

            loss_match = (F.cross_entropy(pseudo_z1[h_i], y_sam)).mean() + (
                F.cross_entropy(pseudo_z2[h_i], y_sam)).mean()

            kl_loss = pq_loss_func(z_fuse_cluster, centers)
            # kl_loss = pq_loss_func(z, centers)

            # total loss
            total_loss = contra_loss + args.beta * loss_match + args.theta * kl_loss

            # total_loss = args.alpha * contra_loss + args.beta * loss_match
            # total_loss = args.alpha * contra_loss + args.theta * kl_loss
            # total_loss = kl_loss
            # kl_loss.backward()

        else:
            total_loss = contra_loss

        total_loss.backward()

        optimizer.step()
        optimizer_gcn_encoder.step()
        # optimizer_lmf.step()
        optimizer_linear.step()
        # optimizer_fuse_linear.step()

        # test stage
        if epoch % 1 == 0:
            model.eval()
            linear.eval()
            # fuse_linear.eval()
            # model_lmf.eval()
            z1 = z * (1 - args.gama) + args.gama * h1
            z2 = z * (1 - args.gama) + args.gama * h2
            hidden_emb = (z1 + z2) / 2

            # hidden_emb = model_lmf(z1, z2)
            # cat = torch.cat((z1, z2), dim=1)
            # hidden_emb = fuse_linear(cat)

            acc, nmi, ari, f1, predict_labels, centers, dis = clustering(hidden_emb, true_labels, args.cluster_num)
            # acc, nmi, ari, f1, predict_labels, centers, dis = clustering(z, true_labels, args.cluster_num)
            # acc_z1, nmi_z1, ari_z1, f1_z1, _, _, _ = clustering(h1, true_labels, args.cluster_num)
            # acc_z2, nmi_z2, ari_z2, f1_z2, _, _, _ = clustering(h2, true_labels, args.cluster_num)
            # acc_z, nmi_z, ari_z, f1_z, _, _, _ = clustering(z, true_labels, args.cluster_num)
            # acc_list2.append(acc)
            # acc_list2 = np.array(acc_list2)
            if acc >= best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1
                best_embed = hidden_emb
            # elif acc_z >= best_acc_z:
            #     best_acc_z = acc_z
            #     best_nmi_z = nmi_z
            #     best_ari_z = ari_z
            #     best_f1_z = f1_z

            # elif acc_z1 >= best_acc_z1:
            #     best_acc_z1 = acc_z1
            #     best_nmi_z1 = nmi_z1
            #     best_ari_z1 = ari_z1
            #     best_f1_z1 = f1_z1
            # elif acc_z2 >= best_acc_z2:
            #     best_acc_z2 = acc_z2
            #     best_nmi_z2 = nmi_z2
            #     best_ari_z2 = ari_z2
            #     best_f1_z2 = f1_z2
        if epoch == args.epochs - 1:
            t_sne(best_embed.detach().cpu().numpy(), true_labels, features.shape[0], True, str(seed))

    end = time.time()
    dt = end - start
    dt_list.append(dt)
    acc_list.append(best_acc)
    nmi_list.append(best_nmi)
    ari_list.append(best_ari)
    f1_list.append(best_f1)

    # acc_list_z.append(best_acc_z)
    # nmi_list_z.append(best_nmi_z)
    # ari_list_z.append(best_ari_z)
    # f1_list_z.append(best_f1_z)

    # acc_list_z1.append(best_acc_z1)
    # nmi_list_z1.append(best_nmi_z1)
    # ari_list_z1.append(best_ari_z1)
    # f1_list_z1.append(best_f1_z1)
    #
    # acc_list_z2.append(best_acc_z2)
    # nmi_list_z2.append(best_nmi_z2)
    # ari_list_z2.append(best_ari_z2)
    # f1_list_z2.append(best_f1_z2)

    tqdm.write("Optimization Finished!")
    tqdm.write('best_acc: {}, best_nmi: {}, best_ari: {}, best_f1: {}'.format(best_acc, best_nmi, best_ari, best_f1))
    # tqdm.write('best_acc_z1: {}, best_nmi_z1: {}, best_ari_z1: {}, best_f1_z1: {}'.format(best_acc_z1, best_nmi_z1, best_ari_z1, best_f1_z1))
    # tqdm.write('best_acc_z2: {}, best_nmi_z2: {}, best_ari_z2: {}, best_f1_z2: {}'.format(best_acc_z2, best_nmi_z2,
    #                                                                                       best_ari_z2, best_f1_z2))
    # tqdm.write('best_acc_z: {}, best_nmi_z: {}, best_ari_z: {}, best_f1_z: {}'.format(best_acc_z, best_nmi_z,
    #                                                                                       best_ari_z, best_f1_z))

acc_list = np.array(acc_list)
nmi_list = np.array(nmi_list)
ari_list = np.array(ari_list)
f1_list = np.array(f1_list)

# acc_list_z = np.array(acc_list_z)
# nmi_list_z = np.array(nmi_list_z)
# ari_list_z = np.array(ari_list_z)
# f1_list_z = np.array(f1_list_z)

# acc_list_z1 = np.array(acc_list_z1)
# nmi_list_z1 = np.array(nmi_list_z1)
# ari_list_z1 = np.array(ari_list_z1)
# f1_list_z1 = np.array(f1_list_z1)
#
# acc_list_z2 = np.array(acc_list_z2)
# nmi_list_z2 = np.array(nmi_list_z2)
# ari_list_z2 = np.array(ari_list_z2)
# f1_list_z2 = np.array(f1_list_z2)

dt_list = np.array(dt_list)
print("ACC ", acc_list.mean(), "±", acc_list.std())
print("NMI ", nmi_list.mean(), "±", nmi_list.std())
print("ARI ", ari_list.mean(), "±", ari_list.std())
print("F1 ", f1_list.mean(), "±", f1_list.std())

# print("ACC_z ", acc_list_z.mean(), "±", acc_list_z.std())
# print("NMI_z ", nmi_list_z.mean(), "±", nmi_list_z.std())
# print("ARI_z ", ari_list_z.mean(), "±", ari_list_z.std())
# print("F1_z ", f1_list_z.mean(), "±", f1_list_z.std())

# print("ACC_z1 ", acc_list_z1.mean(), "±", acc_list_z1.std())
# print("NMI_z1 ", nmi_list_z1.mean(), "±", nmi_list_z1.std())
# print("ARI_z1 ", ari_list_z1.mean(), "±", ari_list_z1.std())
# print("F1_z1 ", f1_list_z1.mean(), "±", f1_list_z1.std())
#
# print("ACC_z2 ", acc_list_z2.mean(), "±", acc_list_z2.std())
# print("NMI_z2 ", nmi_list_z2.mean(), "±", nmi_list_z2.std())
# print("ARI_z2 ", ari_list_z2.mean(), "±", ari_list_z2.std())
# print("F1_z2 ", f1_list_z2.mean(), "±", f1_list_z2.std())
print("运行时间:", dt_list.mean(), "±", dt_list.std())