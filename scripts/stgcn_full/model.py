import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=9, t_stride=1, t_padding=4, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            bias=bias
        )

    def forward(self, x, A):
        x = self.conv(x)  # (N, outC*K, T, V)
        N, KC, T, V = x.size()
        K = A.size(0)
        x = x.view(N, K, KC // K, T, V)  # (N, K, outC, T, V)
        x = torch.einsum("nkctv,kvw->nctw", x, A)
        return x.contiguous()

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, dropout=0.2):
        super().__init__()
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        K = A.shape[0]

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size=K)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9,1), padding=(4,0), stride=(stride,1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x, self.A)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)

class STGCN(nn.Module):
    def __init__(self, num_class, in_channels, A, num_joints=21):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.l1 = STGCNBlock(in_channels, 64, A, residual=False)
        self.l2 = STGCNBlock(64, 64, A)
        self.l3 = STGCNBlock(64, 64, A)
        self.l4 = STGCNBlock(64, 128, A, stride=2)
        self.l5 = STGCNBlock(128, 128, A)
        self.l6 = STGCNBlock(128, 256, A, stride=2)
        self.l7 = STGCNBlock(256, 256, A)

        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (N,V,C,T)
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()  # (N,C,T,V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        x = x.mean(dim=2).mean(dim=2)  # GAP over T and V
        return self.fc(x)