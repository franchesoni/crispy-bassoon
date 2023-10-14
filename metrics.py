import os
from pathlib import Path
import json

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

def aggregate_metrics(per_image_res, clicks_per_image=None):  # we add a [0] because of how we saved the data
  if clicks_per_image is None:
    tps = [res['tps'][0] for res in per_image_res]
    fps = [res['fps'][0] for res in per_image_res]
    tns = [res['tns'][0] for res in per_image_res]
    fns = [res['fns'][0] for res in per_image_res]
  else:
    tps = [res[clicks_per_image[ind]]['tps'][0] for ind, res in enumerate(per_image_res)]
    fps = [res[clicks_per_image[ind]]['fps'][0] for ind, res in enumerate(per_image_res)]
    tns = [res[clicks_per_image[ind]]['tns'][0] for ind, res in enumerate(per_image_res)]
    fns = [res[clicks_per_image[ind]]['fns'][0] for ind, res in enumerate(per_image_res)]
  metrics = compute_global_metrics(tps, fps, tns, fns)
  return metrics


def get_global_click_metrics_from_iis(iis_run_path, max_clicks):
    datapath = Path(iis_run_path) 
    per_image_res = []
    for imgindex in sorted(os.listdir(datapath)):
        with open (datapath / imgindex / 'metrics.json') as f:
            res = json.load(f)
        per_image_res.append(res)

    click_seq = []
    clicks_per_image = [0 for _ in range(len(per_image_res))]
    metrics = [aggregate_metrics(per_image_res, clicks_per_image)]
    for click_n in range(len(per_image_res) * max_clicks):
        print(f'finding click {click_n+1} of {len(per_image_res)*max_clicks}', end='\r')
        best_global_iou = -np.inf  #metrics[-1]['jacc']
        for ind in range(len(clicks_per_image)):  # try clicks on all the images 
            if clicks_per_image[ind] == max_clicks:
                continue
            hypothetical_clicks_per_image = clicks_per_image.copy()
            hypothetical_clicks_per_image[ind] += 1
            hypothetical_metrics = aggregate_metrics(per_image_res, hypothetical_clicks_per_image)
            if hypothetical_metrics['jacc'] > best_global_iou:
                best_global_iou = hypothetical_metrics['jacc']
                best_ind = ind
        clicks_per_image[best_ind] += 1
        click_seq.append(clicks_per_image.copy())
        metrics.append(aggregate_metrics(per_image_res, clicks_per_image))
    return metrics, click_seq
    

