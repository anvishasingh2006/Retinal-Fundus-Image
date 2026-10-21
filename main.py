import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.draw import polygon
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

DRIVE, IDRID, JSIEC = Path("data/DRIVE"), Path("data/IDRiD"), Path("data/JSIEC")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
EPOCHS, POP, ITERS = 100, 20, 40           
PRINTED_EQ45 = True                        
CV_REPEATS = 3                             
np.random.seed(0); torch.manual_seed(0)


def load(p, mask=False):                                  
    im = Image.open(p).convert("L" if mask else "RGB").resize((256, 256), Image.NEAREST if mask else Image.BILINEAR)
    return (np.asarray(im) > 0).astype(np.uint8) if mask else np.asarray(im)  


def prep(rgb):                                         
    return cv2.bilateralFilter(rgb, 9, 75, 75)


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def metrics(y, p):                                        
    y = np.asarray(y); pr = np.asarray(p) > .5
    tp, tn, fp, fn = (pr & (y == 1)).sum(), (~pr & (y == 0)).sum(), (pr & (y == 0)).sum(), (~pr & (y == 1)).sum()
    d = lambda a, b: round(float(a / b) if b else 0., 3)
    return dict(acc=d(tp + tn, len(y)), sens=d(tp, tp + fn), spec=d(tn, tn + fp), ppv=d(tp, tp + fp), npv=d(tn, tn + fn),
                auc=round(float(roc_auc_score(y, p)), 3))


def ogo(fit, lb, ub, pop=20, iters=40, b=1.0, init=None, seed=0):
    """Osprey-Gannet Optimization, Eq. 13-31.: b=1, V1~U(0,1), keep a move only if it improves."""
    rng = np.random.default_rng(seed); lb, ub = np.asarray(lb, float), np.asarray(ub, float); n = lb.size
    Z = lb + rng.random((pop, n)) * (ub - lb)                                
    if init is not None: Z[0] = init
    F = np.array([fit(z) for z in Z]); hist = [F.min()]
    for r in range(1, iters + 1):
        for m in range(pop):
            fish = np.vstack([Z[F < F[m]], Z[F.argmin()][None]])        
            G = fish[rng.integers(len(fish))]
            t, J = rng.random(n), rng.integers(1, 3, n)
            C, V1, Zt = (2 * rng.random() - 1) * b, rng.random(), Z[rng.integers(pop)]
            den = C + t * J; den = np.where(np.abs(den) < 1e-6, 1e-6, den)
            Z1 = np.clip(((C * Zt - V1) * (1 - t * J) + t * G * (1 + C)) / den, lb, ub)   
            f1 = fit(Z1)
            if f1 < F[m]: Z[m], F[m] = Z1, f1
            Z2 = np.clip(Z[m] + (lb + rng.random(n) * (ub - lb)) / r, lb, ub)  
            f2 = fit(Z2)
            if f2 < F[m]: Z[m], F[m] = Z2, f2
        hist.append(F.min())
    return Z[F.argmin()].copy(), hist


def jaya_ccboa(fit, lb, ub, pop=20, iters=40, init=None, seed=0):
    """Jaya Chronological Chef-Based Optimization, Eq. 34-45."""
    rng = np.random.default_rng(seed); lb, ub = np.asarray(lb, float), np.asarray(ub, float); n = lb.size
    Q = lb + rng.random((pop, n)) * (ub - lb)                                
    if init is not None: Q[0] = init
    F = np.array([fit(q) for q in Q]); Qp = Q.copy(); k = max(2, pop // 10); hist = [F.min()]
    for _ in range(iters):
        o = np.argsort(F); Q, F, Qp = Q[o], F[o], Qp[o]                     
        best, worst = Q[0].copy(), Q[-1].copy()
        for l in range(pop):
            if l < k:                                                         
                new = Q[l] + rng.random(n) * (best - rng.integers(1, 3, n) * Q[l])
            else:                                                            
                r1, r2, rr, T = rng.random(n), rng.random(n), rng.uniform(-1, 1, n), rng.uniform(1, 2, n)
                TK = Q[rng.integers(k)]; A, B = 1 - r1 + r2, 1 - rr * T; den = 2 * A - B
                den = np.where(np.abs(den) < 1e-6, 1e-6, den)
                new = ((r2 * worst - r1 * best) * B + A * Qp[l] * B ** 2 + (1 if PRINTED_EQ45 else A) * rr * T * TK * (3 - rr * T)) / den
            new = np.clip(new, lb, ub); fn = fit(new); Qp[l] = Q[l]
            if fn < F[l]: Q[l], F[l] = new, fn
        hist.append(F.min())
    return Q[F.argmin()].copy(), hist



DEFAULT = (0.01, 0.1, 0.0, 1.0)            


def centre(rgb):                            
    g = rgb[..., :2].mean(2) / 255.; s = gaussian(g, 6) * ndi.binary_erosion(g > .08, iterations=10)
    return tuple(int(v) for v in np.unravel_index(s.argmax(), s.shape))


def snake(rgb, p, c):
    img = gaussian(rgb[..., :2].mean(2) / 255., 2)                           
    th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    init = np.stack([c[0] + 23 * np.sin(th), c[1] + 23 * np.cos(th)], 1)
    s = active_contour(img, init, alpha=p[0], beta=p[1], w_line=p[2], w_edge=p[3], gamma=0.01, max_num_iter=150)
    m = np.zeros(img.shape, np.uint8); m[polygon(s[:, 0], s[:, 1], img.shape)] = 1
    return m


def stage_od():
    data = {"train": [], "test": []}
    for gt in sorted(IDRID.rglob("*_OD.tif")):
        img = next(IDRID.rglob(gt.name.replace("_OD.tif", ".jpg")), None)
        if img:
            r = prep(load(img)); data["train" if "train" in str(gt.relative_to(IDRID)).lower() else "test"].append((r, load(gt, True), centre(r)))
    train, test = data["train"][:20], data["test"]
    assert train and test and train[0][1].sum() > 0, "IDRiD OD masks missing or empty after loading - check the mask files"
    print(f"[OD] {len(data['train'])} train / {len(test)} test images")
    win = lambda m, c: m[max(c[0] - 32, 0):c[0] + 32, max(c[1] - 32, 0):c[1] + 32].astype(float)   
    fit = lambda p: float(np.mean([((win(snake(r, p, c), c) - win(g, c)) ** 2).mean() for r, g, c in train]))
    dice = lambda p: float(np.mean([2 * (snake(r, p, c) & g).sum() / (snake(r, p, c).sum() + g.sum() + 1e-9) for r, g, c in test]))
    best, hist = ogo(fit, [0, 0, -.5, 0], [.5, 5, 1, 3], pop=8, iters=8, init=np.array(DEFAULT))   
    print(f"[OD] test Dice  default weights {dice(DEFAULT):.3f}  ->  OGO-tuned {dice(best):.3f}   weights {np.round(best, 3)}")
    json.dump({"params": best.tolist()}, open(OUT / "od_params.json", "w"))
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))               
    for a_, (r, g, c) in zip(ax.ravel(), test[:6]):
        m = snake(r, best, c); a_.imshow(r); a_.contour(g, [.5], colors="lime"); a_.contour(m, [.5], colors="yellow"); a_.axis("off")
        a_.set_title(f"Dice {2 * (m & g).sum() / (m.sum() + g.sum() + 1e-9):.2f}  (green = truth, yellow = snake)", fontsize=9)
    plt.savefig(OUT / "od.png", dpi=110)



class Blk(nn.Module):                      
    def __init__(s, i, o):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.Dropout2d(.1),
                            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o))
        s.skip = nn.Conv2d(i, o, 1)

    def forward(s, x): return torch.relu(s.f(x) + s.skip(x))


class ResUNet(nn.Module):                   
    def __init__(s, f=32):                
        super().__init__()
        s.e1, s.e2, s.b, s.d2, s.d1 = Blk(3, f), Blk(f, 2 * f), Blk(2 * f, 4 * f), Blk(6 * f, 2 * f), Blk(3 * f, f)
        s.pool, s.up, s.head = nn.MaxPool2d(2), nn.Upsample(scale_factor=2), nn.Conv2d(f, 2, 1)

    def feats(s, x):
        e1 = s.e1(x); e2 = s.e2(s.pool(e1)); b = s.b(s.pool(e2))
        return s.d1(torch.cat([s.up(s.d2(torch.cat([s.up(b), e2], 1))), e1], 1))

    def forward(s, x): return s.head(s.feats(x))


def stage_bvs():
    X, Y, M = [], [], []                   
    for f in sorted((DRIVE / "training" / "images").glob("*.tif")):
        n = f.stem.split("_")[0]
        X.append(prep(load(f))); Y.append(load(next((DRIVE / "training" / "1st_manual").glob(n + "_*")), True))
        M.append(load(next((DRIVE / "training" / "mask").glob(n + "_*")), True))
    X = torch.tensor(np.stack(X)).permute(0, 3, 1, 2).float() / 255; Y = torch.tensor(np.stack(Y)).long(); M = np.stack(M).astype(bool)
    idx = np.random.permutation(len(X)); va, tr = idx[:4], idx[4:]
    print(f"[BVS] {len(tr)} train / {len(va)} validation images")
    net = ResUNet(); opt = torch.optim.Adam(net.parameters(), 1e-3); CW = [1., 4.]; ce = nn.CrossEntropyLoss(weight=torch.tensor(CW))
    for ep in range(EPOCHS):                
        net.train()
        for b in np.array_split(np.random.permutation(tr), 4):
            loss = ce(net(X[b]), Y[b]); opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0: print(f"  epoch {ep} loss {loss.item():.3f}")

    def evaluate():
        net.eval()
        with torch.no_grad(): P = torch.softmax(net(X[va]), 1)[:, 1].numpy()
        g, f = Y[va].numpy(), M[va]; m = metrics(g[f], P[f])
        m["dice"] = round(float(2 * ((P > .5) & (g == 1) & f).sum() / (((P > .5) & f).sum() + ((g == 1) & f).sum())), 3)
        return m, P
    print("[BVS] validation, Adam only        :", evaluate()[0])

    net.eval()                             
    with torch.no_grad(): F = net.feats(X[tr]).numpy()
    rng = np.random.default_rng(0); Fp, Tp = [], []
    for i, k in enumerate(tr):
        ys, xs = np.nonzero(M[k]); s = rng.choice(len(ys), 20000, replace=False)
        Fp.append(F[i][:, ys[s], xs[s]].T); Tp.append(Y[k].numpy()[ys[s], xs[s]])
    Fp, Tp = np.concatenate(Fp).astype(np.float32), np.concatenate(Tp); c = Fp.shape[1]
    th0 = np.r_[net.head.weight.detach().numpy().ravel(), net.head.bias.detach().numpy()]
    T1, w = np.eye(2)[Tp], np.where(Tp == 1, CW[1], CW[0])

    def fit(th):                           
        th = th.astype(np.float32); s = softmax(Fp @ th[:-2].reshape(2, c).T + th[-2:])
        return float((w[:, None] * (T1 - s) ** 2).mean())
    best, hist = jaya_ccboa(fit, th0 - .5, th0 + .5, POP, ITERS, init=th0)
    with torch.no_grad():
        net.head.weight.copy_(torch.tensor(best[:-2]).reshape(2, c, 1, 1).float()); net.head.bias.copy_(torch.tensor(best[-2:]).float())
    m, P = evaluate()
    print("[BVS] validation, Adam + Jaya-CCBOA:", m)
    torch.save(net.state_dict(), OUT / "resunet.pt")
    fig, ax = plt.subplots(3, 4, figsize=(12, 9))
    for j in range(4):
        for r, im in enumerate([X[va[j]].permute(1, 2, 0).numpy(), Y[va[j]].numpy(), P[j] * M[va[j]]]):
            ax[r, j].imshow(im, cmap="gray" if r else None); ax[r, j].axis("off")
    ax[0, 0].set_title("input"); ax[1, 0].set_title("ground truth"); ax[2, 0].set_title("Res-UNet vessel probability")
    plt.savefig(OUT / "bvs.png", dpi=110)


RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
sh = lambda g, dy, dx: np.roll(g, (-dy, -dx), (0, 1))        


def stats6(v):                              
    v = np.asarray(v, float).ravel()
    if v.size == 0: return [0.] * 6
    mu = v.mean(); sd = v.std()
    return [mu, v.var(), sd, ((v - mu) ** 3).mean() / (sd ** 3 + 1e-12), ((v - mu) ** 4).mean() / (sd ** 4 + 1e-12), np.sqrt((v ** 2).sum())]


def ldtp(g):                                
    nb = np.stack([sh(g, *d) - g for d in RING]); d = np.abs(nb).argmax(0)
    q = lambda x: np.where(np.abs(x) <= 5, 0, np.where(x > 0, 1, 2))
    return 16 * d + 4 * q(np.take_along_axis(nb, d[None], 0)[0]) + q(np.take_along_axis(nb, ((d + 4) % 8)[None], 0)[0])


def phog(g):                                
    gy, gx = np.gradient(g); mag = np.hypot(gx, gy); b = np.minimum((np.arctan2(gy, gx) % np.pi / np.pi * 8).astype(int), 7)
    rows = []
    for n in (1, 2, 4):
        for i in range(n):
            for j in range(n):
                s = (slice(i * 256 // n, (i + 1) * 256 // n), slice(j * 256 // n, (j + 1) * 256 // n))
                h = np.bincount(b[s].ravel(), weights=mag[s].ravel(), minlength=8); rows.append(h / (h.sum() + 1e-12))
    return np.array(rows)


def lvp(g):                               
    v = g - sh(g, 0, 1)
    return sum(((sh(v, *d) - v >= 0).astype(int) << k) for k, d in enumerate(RING))


def flbp(g):                              
    code, W = 0, 1.
    for k, d in enumerate(RING):
        mu = np.clip(.5 + (sh(g, *d) - g) / 10., 0, 1); code = code + ((mu >= .5).astype(int) << k); W = W * np.maximum(mu, 1 - mu)
    return code * W


def glcm_feats(rgb):                        
    g, out = rgb2gray(rgb) * 255., []
    for m in (ldtp(g), phog(g), lvp(g), flbp(g)):
        m = ((m - m.min()) / (np.ptp(m) + 1e-12) * 31).astype(np.uint8)
        P = graycomatrix(m, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 32, symmetric=True, normed=True)
        out += [float(graycoprops(P, "homogeneity").mean()), float((-(P[:, :, 0] * np.log2(P[:, :, 0] + 1e-12)).sum(axis=(0, 1))).mean())]
    return out


def tanimoto(a, b):                        
    ab = (a * b).sum(); return ab / ((a * a).sum() + (b * b).sum() - ab + 1e-12)


def select(X, thr=.9):                      
    keep = []
    for j in range(X.shape[1]):
        if X[:, j].std() > 1e-9 and all(tanimoto(X[:, j], X[:, k]) < thr for k in keep): keep.append(j)
    return keep


class FeatCNN(nn.Module):                   
    def __init__(s, L):                     
        super().__init__()
        conv = lambda i, o, pool: [nn.Conv1d(i, o, 3, padding=1), nn.ReLU(), nn.Dropout(.2), nn.BatchNorm1d(o)] + ([nn.MaxPool1d(2, ceil_mode=True)] if pool else [])
        dense = lambda i, o: [nn.Linear(i, o), nn.ReLU(), nn.Dropout(.2), nn.BatchNorm1d(o)]
        for _ in range(4): L = (L + 1) // 2                    
        s.body = nn.Sequential(*conv(1, 32, 0), *conv(32, 64, 1), *conv(64, 128, 1), *conv(128, 128, 1), *conv(128, 128, 1),
                               nn.Flatten(), nn.Dropout(.2), *dense(128 * L, 256), *dense(256, 128))   
        s.out = nn.Linear(128, 2)        

    def forward(s, x): return s.out(s.body(x))


def fit_model(Xtr, ytr, seed=0):                    
    lo, hi = Xtr.min(0), Xtr.max(0); sc = lambda Z: np.clip((Z - lo) / (hi - lo + 1e-9), 0, 1)
    torch.manual_seed(seed); keep = select(sc(Xtr)); Xs = torch.tensor(sc(Xtr)[:, keep], dtype=torch.float32)[:, None, :]; yt = torch.tensor(ytr)
    cw = len(ytr) / (2 * np.bincount(ytr)); net = FeatCNN(len(keep)); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32))
    for _ in range(150):
        net.train()
        for b in torch.randperm(len(Xs)).split(16):
            if len(b) > 1: loss = ce(net(Xs[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): H = net.body(Xs).numpy()
    th0 = np.r_[net.out.weight.detach().numpy().ravel(), net.out.bias.detach().numpy()]
    T1 = np.eye(2)[ytr]
    fit = lambda th: float(((T1 - softmax(H @ th[:-2].reshape(2, -1).T + th[-2:])) ** 2).mean())  
    th, hist = ogo(fit, th0 - .5, th0 + .5, POP, ITERS, init=th0, seed=seed)
    def predict(Z, th):
        with torch.no_grad(): Hz = net.body(torch.tensor(sc(Z)[:, keep], dtype=torch.float32)[:, None, :]).numpy()
        return softmax(Hz @ th[:-2].reshape(2, -1).T + th[-2:])[:, 1]
    return predict, th0, th, hist, net


def stage_cls():
    net = ResUNet(); net.load_state_dict(torch.load(OUT / "resunet.pt", map_location="cpu")); net.eval()
    par = json.load(open(OUT / "od_params.json"))["params"] if (OUT / "od_params.json").exists() else DEFAULT
    items = []
    for d in sorted(p for p in JSIEC.rglob("*") if p.is_dir()):
        lab = 0 if "normal" in d.name.lower() else 1 if "hypertensive" in d.name.lower() else None
        if lab is not None: items += [(f, lab) for f in sorted(d.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif"}]
    y = np.array([l for _, l in items]); print(f"[CLS] {int((y == 0).sum())} normal, {int((y == 1).sum())} hypertensive images")
    X = []
    for f, _ in items:                      
        raw = load(f); r = prep(raw); od = snake(r, par, centre(r))
        with torch.no_grad(): pv = torch.softmax(net(torch.tensor(r).permute(2, 0, 1)[None].float() / 255), 1)[0, 1].numpy()
        X.append(stats6(r[..., 1][od > 0]) + stats6(pv[r.mean(2) > 20]) + glcm_feats(raw))
    X = np.array(X); names = ["CNN (Adam only)", "CNN + OGO output layer"]

    def cv(X, y, seed):              
        oof = np.zeros((2, len(y)))
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
            predict, th0, th, _, _ = fit_model(X[tr], y[tr], seed * 10 + k); oof[0, te], oof[1, te] = predict(X[te], th0), predict(X[te], th)
        return oof
    runs = [cv(X, y, s_) for s_ in range(CV_REPEATS)]
    for i, n in enumerate(names):
        ms = [metrics(y, o[i]) for o in runs]
        print(f"[CLS] {n:24s}", {k: f"{np.mean([m[k] for m in ms]):.3f}+-{np.std([m[k] for m in ms]):.3f}" for k in ms[0]}, f"({CV_REPEATS} CV repeats)")
    y_shuf = np.random.default_rng(0).permutation(y)        
    print("[CLS] label-shuffle check, AUC =", metrics(y_shuf, cv(X, y_shuf, 0)[0])["auc"], "(about 0.5 expected; much higher would mean leakage)")
    plt.figure(figsize=(5, 5))
    for n, p in zip(names, runs[0]): fpr, tpr, _ = roc_curve(y, p); plt.plot(fpr, tpr, label=n)
    plt.plot([0, 1], [0, 1], "k:"); plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("out-of-fold ROC (repeat 0)"); plt.savefig(OUT / "cls_roc.png", dpi=110)
    predict, th0, th, hist, cnn = fit_model(X, y); torch.save({"cnn": cnn.state_dict(), "ogo_head": th}, OUT / "classifier.pt")


if __name__ == "__main__":
    if IDRID.exists(): stage_od()
    else: print("skip OD stage: data/IDRiD not found")
    if DRIVE.exists(): stage_bvs()
    else: print("skip BVS stage: data/DRIVE not found")
    if JSIEC.exists() and (OUT / "resunet.pt").exists(): stage_cls()
    else: print("skip classification: need data/JSIEC and outputs/resunet.pt (run the BVS stage first)")