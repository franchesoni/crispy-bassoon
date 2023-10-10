
import numpy as np

"""### Metrics"""

def compute_tps_fps_tns_fns(preds, masks):
  tps, fps, tns, fns = [], [], [], []
  for pred, mask in zip(preds, masks):
    if pred is None:
      pred = np.zeros_like(mask)
    label = 0
    fns.append(((pred == label) * (mask != label)).sum())
    tns.append(((pred == label) * (mask == label)).sum())
    label = 1
    fps.append(((pred == label) * (mask != label)).sum())
    tps.append(((pred == label) * (mask == label)).sum())
  return tps, fps, tns, fns

def compute_global_metrics(tps, fps, tns, fns):
  # average statistics
  accs, jaccs = [], []
  for i in range(len(tps)):
    acc = (tps[i] + tns[i]) / (tps[i] + tns[i] + fps[i] + fns[i])  # accuracy
    jacc = (tps[i]) / (tps[i] + fps[i] + fns[i])  # jaccard index
    accs.append(acc)
    jaccs.append(jacc)
  avg_acc = sum(accs) / len(accs) if len(accs) > 0 else 0
  avg_jacc = sum(jaccs) / len(jaccs) if len(jaccs) > 0 else 0  


  # global statistics (everything as one big image)
  gtps, gfps, gtns, gfns = sum(tps), sum(fps), sum(tns), sum(fns)
  acc = (gtps + gtns) / (gtps + gtns + gfps + gfns)  # accuracy
  jacc = (gtps) / (gtps + gfps + gfns)  # jaccard index
  return {'acc': acc, 'jacc': jacc, 'avg_acc': avg_acc, 'avg_jacc': avg_jacc, 'tps': tps, 'fps': fps, 'tns': tns, 'fns': fns}