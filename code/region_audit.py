"""Regional map metrics and perturbation controls."""
import numpy as np

def region_metrics(heatmap, mask, fraction=.10):
    heat=np.asarray(heatmap,dtype=float); target=np.asarray(mask,dtype=bool)
    if heat.ndim!=2 or heat.shape!=target.shape:
        raise ValueError('Map and annotation must share the SAME pixel frame and shape.')
    if not np.isfinite(heat).all() or (heat<0).any() or not 0<fraction<=1:
        raise ValueError('Require finite, nonnegative attribution and fraction in (0,1].')
    if not target.any():
        return {'status':'no_positive_region','pointing_hit':None,'attribution_mass_in_region':None,'iou':None}
    if heat.sum()==0 or np.ptp(heat)==0:
        return {'status':'uninformative_map','pointing_hit':None,'attribution_mass_in_region':None,'iou':None}
    k=max(1,int(np.ceil(fraction*heat.size)))

    order=np.argsort(-heat.ravel(),kind='stable')
    selected=np.zeros(heat.size,dtype=bool);selected[order[:k]]=True;selected=selected.reshape(heat.shape)
    return {'status':'evaluated','pointing_hit':bool(target.ravel()[order[0]]),
            'attribution_mass_in_region':float(heat[target].sum()/heat.sum()),
            'iou':float((selected&target).sum()/(selected|target).sum()),
            'selected_fraction':k/heat.size,'annotated_fraction':float(target.mean())}

def perturbation_audit(image, heatmap, predict, seed=20261008, repetitions=20, grid=8, fraction=.10):
    """Model score changes after equal-area cell removal; illustrative faithfulness audit.

    predict accepts ONE RGB array in the declared model-input frame and returns
    the same calibrated positive-class score used for the original image.
    Compare attributed cells with matched random cells under two fill schemes.
    Caller must use held-out images and retain source groups for uncertainty.
    """
    from PIL import Image, ImageFilter
    image=np.asarray(image,dtype=np.uint8);heat=np.asarray(heatmap,dtype=float)
    if image.ndim!=3 or image.shape[2]!=3 or heat.shape!=image.shape[:2]:
        raise ValueError('RGB image and attribution must share the SAME model-input frame.')
    if not np.isfinite(heat).all() or np.ptp(heat)==0 or (heat<0).any():
        raise ValueError('Require an informative finite nonnegative attribution map.')
    if not 0<fraction<=1 or repetitions<1 or grid<1 or grid>min(heat.shape):
        raise ValueError('Invalid perturbation configuration.')
    baseline=float(predict(image))
    if not np.isfinite(baseline) or not 0<=baseline<=1:
        raise ValueError('predict must return a positive-class probability.')
    ys=np.linspace(0,image.shape[0],grid+1,dtype=int);xs=np.linspace(0,image.shape[1],grid+1,dtype=int)
    cells=[(slice(ys[y],ys[y+1]),slice(xs[x],xs[x+1])) for y in range(grid) for x in range(grid)]
    k=max(1,int(np.ceil(fraction*len(cells))))
    scores=[float(heat[c].mean()) for c in cells]
    top=np.argsort(-np.asarray(scores),kind='stable')[:k]
    rng=np.random.default_rng(seed)

    byarea={}
    for i,c in enumerate(cells):byarea.setdefault(image[c].shape[:2],[]).append(i)
    def random_control():
        selected=[]
        for shape in sorted({image[cells[i]].shape[:2] for i in top}):
            n=sum(image[cells[i]].shape[:2]==shape for i in top)
            selected.extend(rng.choice(byarea[shape],size=n,replace=False).tolist())
        return selected
    controls=[random_control() for _ in range(repetitions)]
    fills={'mean':np.broadcast_to(image.mean(axis=(0,1)).astype(np.uint8),image.shape),
           'blur':np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=max(image.shape[:2])/16)))}
    rows=[]
    def score(indices,fill):
        changed=image.copy()
        for i in indices:changed[cells[i]]=fill[cells[i]]
        value=float(predict(changed))
        if not np.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid perturbed probability.')
        return value
    for name,fill in fills.items():
        attributed_drop=baseline-score(top,fill)
        drops=np.array([baseline-score(control,fill) for control in controls])
        rows.append({'fill':name,'original_probability':baseline,'attributed_score_drop':attributed_drop,
                     'random_mean_score_drop':float(drops.mean()),'excess_drop_over_random':float(attributed_drop-drops.mean()),
                     'random_score_drops':drops.tolist(),'cells_removed':k,'random_repetitions':repetitions,
                     'interpretation':'Perturbation sensitivity; not proof of causal or forensic validity.'})
    return rows
