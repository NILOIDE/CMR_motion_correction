"""Does torch optimizer <name> give identical results replayed as a CUDA graph (with tensor learning rates updated in
place between replays) and run eagerly? Usage: python opt_graph_test.py <name>"""
import sys
import torch

dev = torch.device("cuda")


def run(name, graph):
    torch.manual_seed(0)
    a = torch.randn(14, 3, device=dev).requires_grad_(True)
    b = torch.randn(14, 3, device=dev).requires_grad_(True)
    lr_a, lr_b = torch.tensor(0.2, device=dev), torch.tensor(0.04, device=dev)
    groups = [{"params": [a], "lr": lr_a}, {"params": [b], "lr": lr_b}]
    kw = dict(capturable=True)
    opt = {"adam": lambda: torch.optim.Adam(groups, **kw), "adamw": lambda: torch.optim.AdamW(groups, weight_decay=0.0, **kw),
           "sgd": lambda: torch.optim.SGD(groups, momentum=0.9), "sgd_foreach": lambda: torch.optim.SGD(groups, momentum=0.9, foreach=True),
           "sgd_nesterov": lambda: torch.optim.SGD(groups, momentum=0.9, nesterov=True),
           "rmsprop": lambda: torch.optim.RMSprop(groups, **kw), "radam": lambda: torch.optim.RAdam(groups, **kw),
           "nadam": lambda: torch.optim.NAdam(groups, **kw), "adamax": lambda: torch.optim.Adamax(groups, **kw),
           "adagrad": lambda: torch.optim.Adagrad(groups)}[name]()

    def body():
        opt.zero_grad(set_to_none=False)
        loss = (torch.sin(a) * a).sum() + (b.pow(2) * torch.cos(b)).sum()
        loss.backward()
        opt.step()

    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                body()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):           # recorded, not executed
            body()
    else:
        for _ in range(2):
            body()
    for k in range(20):
        lr_a.fill_(0.2 * (1 - k / 20))
        lr_b.fill_(0.04 * (1 - k / 20))
        g.replay() if graph else body()
    torch.cuda.synchronize()
    return torch.cat([a.detach().flatten(), b.detach().flatten()])


name = sys.argv[1]
try:
    e, g = run(name, False), run(name, True)
    print(f"{name:13s} graph vs eager max diff {float((e - g).abs().max()):.2e}")
except Exception as ex:
    print(f"{name:13s} FAILED: {type(ex).__name__}: {str(ex).splitlines()[0][:120]}")
