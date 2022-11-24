import torch
import subprocess

def score(trainer, X, device):
    net = trainer.ae_net
    with torch.no_grad():
        X = torch.FloatTensor(X).to(device)
        y = net(X)
        dist = torch.sum((y - X)**2, dim=1)
        scores = 100-4*dist
    return scores


# set gpu option
def set_cuda_visible_device(ngpus):
    empty = []
    for i in range(3,4):
        command = ['nvidia-smi','-i',str(i)]
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        result = str(p.communicate()[0])
        count = result.count('No running')
        if count>0:
            empty.append(i)
    if len(empty)<ngpus:
        assert False, f"Available gpus are less than required: ngpus={ngpus}, empty={len(empty)}"
    cmd = ''
    for i in range(ngpus):
        cmd += str(empty[i]) + ','
    return cmd.rstrip(',')
    
