import numpy as np

HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1
    hop_dis = np.full((num_node, num_node), np.inf)
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, axis=0)
    Dn = np.zeros_like(A)
    for i in range(A.shape[0]):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return A @ Dn

class Graph:
    def __init__(self, num_node=21, max_hop=1, dilation=1, strategy="spatial", center=0):
        self.num_node = num_node
        self.self_link = [(i, i) for i in range(num_node)]
        self.neighbor_link = HAND_EDGES
        self.edge = self.self_link + self.neighbor_link
        self.center = center

        self.max_hop = max_hop
        self.dilation = dilation
        self.hop_dis = get_hop_distance(num_node, self.neighbor_link, max_hop=max_hop)
        self.A = self.get_adjacency(strategy)

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1

        adjacency = normalize_digraph(adjacency)

        if strategy == "uniform":
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = adjacency
            return A.astype(np.float32)

        if strategy == "distance":
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = adjacency[self.hop_dis == hop]
            return A.astype(np.float32)

        if strategy == "spatial":
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            dist_j = self.hop_dis[j, self.center]
                            dist_i = self.hop_dis[i, self.center]
                            if dist_j == dist_i:
                                a_root[j, i] = adjacency[j, i]
                            elif dist_j < dist_i:
                                a_close[j, i] = adjacency[j, i]
                            else:
                                a_further[j, i] = adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            return np.stack(A).astype(np.float32)

        raise ValueError(f"Unknown strategy: {strategy}")