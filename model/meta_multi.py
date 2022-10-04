import torch
import torch.nn as nn
import scipy.misc

def make_model(args, parent=False):
    return MSR_RDN(args)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class GenWeights(nn.Module):
    def __init__(self,inpC=64, kernel_size=3, outC=32):
        super(GenWeights,self).__init__()
        self.kernel_size=kernel_size
        self.outC = outC
        self.inpC = inpC
        self.meta_block=nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=(3 - 1) // 2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=(3 - 1) // 2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=(3 - 1) // 2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=(3 - 1) // 2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=(3 - 1) // 2, stride=1)
        )
    def forward(self,x):
        output = self.meta_block(x)
        return output

class FUSE_RDN(nn.Module):
    def __init__(self, args):
        super(FUSE_RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks D, conv layers within the blocks C, out channels G within the last layer of the blocks,
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
            'C': (4, 6, 12),
        }[args.RDNconfig]
        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(2, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G0, 1, kSize, padding=(kSize-1)//2, stride=1)
        ])  

    def forward(self, inp):
        f__1 = self.SFENet1(inp)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1
        return self.UPNet(x)+inp.mean(1).unsqueeze(1)


class MSR_RDN(nn.Module):
    def __init__(self, args):
        super(MSR_RDN, self).__init__()
        G0 = args.G0
        kSize = args.RDNkSize
        self.stage = args.stage
        # number of RDB blocks D, conv layers within the blocks C, out channels G within the last layer of the blocks,
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
            'C': (6, 8, 32)
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(3, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        self.GW = GenWeights()

        self.RFN = FUSE_RDN(args)




    def forward(self, inp_x, dist):
        f__1 = self.SFENet1(inp_x)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        for i in range(int(dist.size(1))):
            if i == 0:
                weights = self.GW(dist[:,[i],:,:]).view(x.size(0),1,64,3,3)
            else:
                weights = torch.cat((weights, self.GW(dist[:,[i],:,:]).view(x.size(0),1,64,3,3)), 1)
        inp = nn.functional.unfold(x, 3, padding=1)
        output = inp.transpose(1, 2).matmul(weights.view(weights.size(0), weights.size(1), -1).transpose(1,2)).transpose(1, 2)
        output = output.view(-1, dist.size(1), x.size(2),x.size(3))
        return output


    def marginal_inference(self, inp_vol, weights):
        holder = []
        for i in range(512):
            f__1 = self.SFENet1(inp_vol[i:i+1])
            # f__1 = self.SFENet1(inp_vol[i*8:(i+1)*8])
            x = self.SFENet2(f__1)
            RDBs_out = []
            for i in range(self.D):
                x = self.RDBs[i](x)
                RDBs_out.append(x)
            x = self.GFF(torch.cat(RDBs_out, 1))
            feat_prev = x + f__1 
            inp = nn.functional.unfold(feat_prev, 3, padding=1)
            # print(inp.shape,weights.view(weights.size(0), weights.size(1), -1).transpose(1,2))
            output = inp.transpose(1, 2).matmul(weights.view(weights.size(0), weights.size(1), -1).transpose(1,2)).transpose(1, 2)
            output = output.view(-1, weights.size(1), x.size(2),x.size(3))
            holder.append(output)

        output = torch.cat(holder)
        comb = []
        for i in range(weights.size(1)):
            comb.append(output[:,i])
        comb = torch.stack(comb,dim=3).view(output.shape[0],512,output.shape[-1]*weights.size(1))#this'd be 64,64,60
        return comb


    def test(self, inp, dist, factor):
        #input in this case should be something like 
        for i in range(factor):
            if i == 0:
                weights = self.GW(dist[:,[i],:,:]).view(1,1,64,3,3)#.expand(8,1,64,3,3)
            else:
                weights = torch.cat((weights, self.GW(dist[:,[i],:,:]).view(1,1,64,3,3)), 1)#.expand(8,1,64,3,3)
        # print(weights.shape)
        inp_sag = inp[0,0]
        inp_cor = inp[0,1]

        comb_sag = self.marginal_inference(inp_sag, weights)
        comb_cor = self.marginal_inference(inp_cor, weights).permute(1,0,2)

        comb_sag, comb_cor = torch.clamp(comb_sag,0,4000).round(), torch.clamp(comb_cor,0,4000).round()

        if self.stage == 1:
            return comb_sag.unsqueeze(0), comb_cor.unsqueeze(0)
        else:
            vol = torch.stack([comb_sag,comb_cor]).permute(3,0,1,2)
            ref = []
            for i in range(vol.shape[0]):
                ref.append(self.RFN(vol[[i]])[0])
            ref = torch.cat(ref).permute(1,2,0)
            return ref.unsqueeze(0), comb_sag.unsqueeze(0), comb_cor.unsqueeze(0)


