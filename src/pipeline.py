"""Reproducible monthly consumption networks, baseline comparisons and validation."""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import igraph as ig
import leidenalg as la
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import kruskal
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from metrics import evaluate, graph_scores

CATEGORIES=['Продовольствие','Здоровье','Маркетплейсы','Общественное питание','Транспорт','Прочие расходы']

def dump(path,value):
    def conv(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        raise TypeError(type(x).__name__)
    # Null denotes an undefined metric, never silently replace it with zero.
    value=json.loads(json.dumps(value,default=conv,ensure_ascii=False))
    def clean(v):
        if isinstance(v,float) and not np.isfinite(v):return None
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        return v
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2))

def prepare(raw):
    file=raw/'consumption.parquet'
    d=pd.read_parquet(file)
    if 'value' not in d and 'consumption' in d:d=d.rename(columns={'consumption':'value'})
    required={'territory_id','date','category','value'}
    if not required.issubset(d.columns):raise ValueError(f'Expected {required}')
    if d[list(required)].isna().any().any():raise ValueError('Null fields in source')
    if d.duplicated(['territory_id','date','category']).any():raise ValueError('Duplicate keys')
    if not np.isfinite(d.value).all() or (d.value<0).any():raise ValueError('Invalid expense')
    dates=sorted(d.date.unique()); ids0=sorted(d.territory_id.unique())
    if list(pd.period_range(dates[0],dates[-1],freq='M').astype(str))!=dates:raise ValueError('Nonconsecutive months')
    if len(dates)<2:raise ValueError('At least two months required')
    expected=set(CATEGORIES[:-1]+['Все категории'])
    if set(d.category.unique())!=expected:raise ValueError('Unexpected categories; update explicit adapter')
    p=d.pivot(index=['date','territory_id'],columns='category',values='value')
    p=p.reindex(pd.MultiIndex.from_product([dates,ids0],names=['date','territory_id']))
    good=p.notna().all(axis=1)&(p['Все категории']>0)
    remainder=p['Все категории']-p[CATEGORIES[:-1]].sum(axis=1,min_count=5)
    good &= remainder>=0
    valid=good.groupby(level='territory_id').all()
    ids=np.array(valid[valid].index)
    if len(ids)<30:raise ValueError('Insufficient complete territories')
    p=p.loc[(slice(None),ids),:]
    p['Прочие расходы']=p['Все категории']-p[CATEGORIES[:-1]].sum(axis=1)
    amounts=np.stack([p.xs(t)[CATEGORIES].reindex(ids).values for t in dates])
    totals=np.stack([p.xs(t)['Все категории'].reindex(ids).values for t in dates])
    shares=amounts/totals[:,:,None]
    if not np.allclose(shares.sum(2),1):raise ValueError('Shares must sum to 1')
    audit=dict(source_rows=len(d),source_territories=len(ids0),included_territories=len(ids),excluded_territories=len(ids0)-len(ids),
               months=len(dates),start=dates[0],end=dates[-1],panel_rows=len(ids)*len(dates),
               excluded_ids=[int(i) for i in valid[~valid].index],missing_policy='complete-case balanced panel; no imputation',
               source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),value_column='value',
               remainder_min=float(p['Прочие расходы'].min()))
    return dates,ids,shares,totals,audit

def build_graph(X,k=12,metric='euclidean'):
    """Union kNN graph; self excluded explicitly, deterministic sorting for ties.

    w_ij=exp(-d_ij²/(sigma_i sigma_j)); sigma_i=distance to kth neighbor.
    Edges represent feature similarity, not payments or commuting.
    """
    n=len(X); k=min(k,n-1)
    nn=NearestNeighbors(n_neighbors=min(n,k+1),metric=metric).fit(X)
    ds,inds=nn.kneighbors(X)
    near=[]; distances=[]
    for i in range(n):
        vals=sorted([(float(max(d,0)),int(j)) for d,j in zip(ds[i],inds[i]) if j!=i],key=lambda x:(x[0],x[1]))[:k]
        near.append([j for d,j in vals]);distances.append([d for d,j in vals])
    sigma=np.maximum([v[-1] for v in distances],1e-12)
    edges={}
    for i,(js,dd) in enumerate(zip(near,distances)):
        for j,d in zip(js,dd):
            e=tuple(sorted((i,j)));edges[e]=float(np.exp(-d*d/(sigma[i]*sigma[j])))
    keys=sorted(edges)
    g=ig.Graph(n=n,edges=keys,directed=False);g.es['weight']=[edges[e] for e in keys]
    return g

def leiden(g,resolution,seed):
    return np.array(la.find_partition(g,la.RBConfigurationVertexPartition,weights='weight',resolution_parameter=resolution,seed=seed,n_iterations=-1).membership)

def align(previous,current,next_id):
    """Maximum-Jaccard one-to-one matching, births receive fresh IDs.

    Matching is a display convention; transition table retains all splits/merges.
    ARI and raw-profile distances are label independent.
    """
    a=np.unique(previous);b=np.unique(current); J=np.zeros((len(a),len(b)))
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            inter=((previous==x)&(current==y)).sum(); union=((previous==x)|(current==y)).sum();J[i,j]=inter/union
    rows,cols=linear_sum_assignment(-J);mapping={}
    for i,j in zip(rows,cols):
        if J[i,j]>0:mapping[int(b[j])]=int(a[i])
    for y in b:
        if int(y) not in mapping:mapping[int(y)]=next_id;next_id+=1
    return np.array([mapping[int(y)] for y in current]),next_id

def fit(cfg):
    start=time.time();raw=Path(cfg['raw_dir']);out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    dates,ids,shares,totals,audit=prepare(raw)
    manifest=raw.parent/'manifest.json'
    if manifest.exists():
        for name,expected in json.loads(manifest.read_text())['files'].items():
            if hashlib.sha256((raw/name).read_bytes()).hexdigest()!=expected:raise ValueError('Source checksum changed: '+name)
    dump(out/'audit.json',audit)
    raw_shares=shares.copy();window=cfg['smoothing_months']
    shares=np.array([raw_shares[max(0,t-window+1):t+1].mean(0) for t in range(len(dates))])
    Xs=np.sqrt(shares);seed=cfg['seed'];neighbors=cfg['neighbors'];sample=cfg['silhouette_sample']
    graphs=[build_graph(X,neighbors) for X in Xs]
    calibration=[]
    for gamma in cfg['resolution_grid']:
        for t in range(window-1,min(window-1+cfg['calibration_months'],len(dates))):
            labels=leiden(graphs[t],gamma,seed);e=evaluate(Xs[t],labels,graphs[t],sample,seed)
            calibration.append(dict(date=dates[t],resolution=gamma,**e))
    cal=pd.DataFrame(calibration);cal.to_csv(out/'calibration.csv',index=False)
    # Select by mean SW subject to minimum size across calibration months.
    valid=cal.groupby('resolution').filter(lambda d:(d.min_size>=cfg['min_cluster_size']).all())
    if valid.empty:raise ValueError('No calibration candidate meets minimum cluster size')
    gamma=float(valid.groupby('resolution').SW.mean().idxmax())
    monthly=[];labels_all=[];transitions=[];profiles=[];rows=[];next_id=0
    projection=PCA(n_components=2).fit(Xs[0]) # Fixed January 2023 basis, no drifting axes.
    coords=np.array([projection.transform(X) for X in Xs])
    for t,(date,X,g) in enumerate(zip(dates,Xs,graphs)):
        print(f'{date} {t+1}/{len(dates)}',flush=True)
        raw_label=leiden(g,gamma,seed);K=len(np.unique(raw_label))
        aligned=raw_label.copy() if t==0 else align(labels_all[-1],raw_label,next_id)[0]
        next_id=max(next_id,int(aligned.max())+1);labels_all.append(aligned)
        methods={'Leiden':raw_label,'KMeans':KMeans(n_clusters=K,n_init=20,random_state=seed).fit_predict(X),
                 'Ward':AgglomerativeClustering(n_clusters=K,linkage='ward').fit_predict(X)}
        for method,y in methods.items():monthly.append(dict(date=date,method=method,**evaluate(X,y,g,sample,seed)))
        for c in np.unique(aligned):
            mask=aligned==c;p=shares[t,mask].mean(0)
            profiles.append(dict(date=date,cluster=int(c),size=int(mask.sum()),**dict(zip(CATEGORIES,p)),
                                  median_total=float(np.median(totals[t,mask]))))
        delta=np.linalg.norm(np.sqrt(raw_shares[t])-np.sqrt(raw_shares[t-1]),axis=1)/np.sqrt(2) if t>0 else np.zeros(len(ids))
        for i,tid in enumerate(ids):rows.append(dict(date=date,territory_id=int(tid),cluster=int(aligned[i]),
                     total=float(totals[t,i]),hellinger_change=float(delta[i]),pc1=float(coords[t,i,0]),pc2=float(coords[t,i,1])))
        if t:
            a=labels_all[-2];b=aligned
            for x in np.unique(a):
                for y in np.unique(b):
                    count=int(((a==x)&(b==y)).sum())
                    if count:transitions.append(dict(from_date=dates[t-1],date=date,from_cluster=int(x),to_cluster=int(y),count=count))
    labels_all=np.array(labels_all)
    stability=[]
    for t in range(1,len(dates)):
        changes=labels_all[t]!=labels_all[t-1]
        stability.append(dict(date=dates[t],ARI=float(adjusted_rand_score(labels_all[t-1],labels_all[t])),
                              switch_share=float(changes.mean()),mean_hellinger=float((np.linalg.norm(Xs[t]-Xs[t-1],axis=1)/np.sqrt(2)).mean()),
                              yoy_ARI=float(adjusted_rand_score(labels_all[t-12],labels_all[t])) if t>=12 else None))
    pd.DataFrame(monthly).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(profiles).to_csv(out/'profiles.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'assignments.csv',index=False)
    pd.DataFrame(transitions).to_csv(out/'transitions.csv',index=False)
    pd.DataFrame(stability).to_csv(out/'temporal_stability.csv',index=False)
    g=graphs[-1];y=labels_all[-1];X=Xs[-1];rng=np.random.default_rng(seed)
    sensitivity=[]
    for metric in ['hellinger','cosine']:
        Z=X if metric=='hellinger' else shares[-1]
        for k in cfg['sensitivity_neighbors']:
            gg=build_graph(Z,k,'euclidean' if metric=='hellinger' else 'cosine')
            yy=leiden(gg,gamma,seed)
            sensitivity.append(dict(similarity=metric,neighbors=k,ARI_to_main=float(adjusted_rand_score(y,yy)),**evaluate(X,yy,gg,sample,seed)))
    pd.DataFrame(sensitivity).to_csv(out/'sensitivity.csv',index=False)
    boot=[];support=np.zeros(len(ids))
    for repeat in range(cfg['bootstrap_repeats']):
        idx=np.sort(rng.choice(len(ids),int(len(ids)*cfg['bootstrap_fraction']),replace=False))
        gg=build_graph(X[idx],neighbors);yy=leiden(gg,gamma,seed+repeat)
        boot.append(dict(repeat=repeat,kind='subsample',ARI=float(adjusted_rand_score(y[idx],yy))))
        yy=leiden(g,gamma,seed+repeat)
        boot.append(dict(repeat=repeat,kind='seed',ARI=float(adjusted_rand_score(y,yy))))
        aligned_seed,_=align(y,yy,int(y.max())+1);support+=(aligned_seed==y)
    pd.DataFrame({'territory_id':ids,'cluster':y,'seed_support':support/cfg['bootstrap_repeats']}).to_csv(out/'confidence_latest.csv',index=False)
    for win in [1,3,6]:
        smooth=raw_shares[max(0,len(dates)-win):].mean(0);Z=np.sqrt(smooth);gg=build_graph(Z,neighbors);yy=leiden(gg,gamma,seed)
        sensitivity.append(dict(similarity='window_'+str(win),neighbors=neighbors,ARI_to_main=float(adjusted_rand_score(y,yy)),**evaluate(Z,yy,gg,sample,seed)))
    pd.DataFrame(sensitivity).to_csv(out/'sensitivity.csv',index=False)
    pd.DataFrame(boot).to_csv(out/'robustness.csv',index=False)
    null=[]
    for repeat in range(cfg['null_repeats']):
        yp=rng.permutation(y) # Exact same cluster sizes, breaks labels relative to graph/features.
        null.append(dict(repeat=repeat,**evaluate(X,yp,g,sample,seed)))
    pd.DataFrame(null).to_csv(out/'random_baseline.csv',index=False)
    # External interpretation only; market_access not used in clustering.
    market=pd.read_parquet(raw/'market_access.parquet').set_index('territory_id').market_access.reindex(ids)
    external=[];ma=market.to_numpy()
    for c in np.unique(y):
        vals=ma[(y==c)&np.isfinite(ma)]
        external.append(dict(cluster=int(c),n=len(vals),median=float(np.median(vals)) if len(vals) else None,
                             q25=float(np.quantile(vals,.25)) if len(vals) else None,q75=float(np.quantile(vals,.75)) if len(vals) else None))
    groups=[ma[(y==c)&np.isfinite(ma)] for c in np.unique(y)];groups=[v for v in groups if len(v)]
    H,pval=kruskal(*groups) if len(groups)>1 else (np.nan,np.nan)
    ext=dict(market_access=external,coverage=int(np.isfinite(ma).sum()),kruskal_H=float(H),kruskal_p=float(pval),
             effect_epsilon_squared=float(max(0,(H-len(groups)+1)/(np.isfinite(ma).sum()-len(groups)))) if len(groups)>1 else None)
    # Road distances: compare final economic edges with random pairs on same observed coverage.
    road=pd.read_parquet(raw/'connection.parquet');road=road[road.type=='highway'].copy()
    road['a']=np.minimum(road.territory_id_x,road.territory_id_y);road['b']=np.maximum(road.territory_id_x,road.territory_id_y)
    road=road[['a','b','distance']].drop_duplicates(['a','b'])
    edges=np.array(g.get_edgelist());ed=pd.DataFrame({'a':np.minimum(ids[edges[:,0]],ids[edges[:,1]]),'b':np.maximum(ids[edges[:,0]],ids[edges[:,1]])})
    near=ed.merge(road,on=['a','b'],how='left').distance
    pairs=rng.integers(0,len(ids),size=(len(edges)*2,2));pairs=pairs[pairs[:,0]!=pairs[:,1]][:len(edges)]
    rand=pd.DataFrame({'a':np.minimum(ids[pairs[:,0]],ids[pairs[:,1]]),'b':np.maximum(ids[pairs[:,0]],ids[pairs[:,1]])}).merge(road,on=['a','b'],how='left').distance
    ext['roads']=dict(economic_edges_total=len(near),economic_edges_observed=int(near.notna().sum()),economic_edges_median_km=float(near.median()),
                      random_pairs_observed=int(rand.notna().sum()),random_pairs_median_km=float(rand.median()))
    dump(out/'external_validation.json',ext)
    np.savez_compressed(out/'panel.npz',ids=ids,shares=shares,raw_shares=raw_shares,totals=totals,labels=labels_all,coords=coords,dates=np.array(dates),categories=np.array(CATEGORIES))
    pd.DataFrame({'source':ids[edges[:,0]],'target':ids[edges[:,1]],'weight':g.es['weight']}).to_csv(out/'edges_latest.csv',index=False)
    summary=dict(audit=audit,selected_resolution=gamma,neighbors=neighbors,
                 smoothing_months=window,calibration_months=dates[window-1:window-1+cfg['calibration_months']],latest_metrics=[r for r in monthly if r['date']==dates[-1]],
                 latest_clusters=int(len(np.unique(y))),core_share=float((support/cfg['bootstrap_repeats']>=.8).mean()),pca_variance_explained=projection.explained_variance_ratio_.tolist(),
                 mean_monthly_ARI=float(np.mean([r['ARI'] for r in stability])),mean_switch_share=float(np.mean([r['switch_share'] for r in stability])),
                 stability_subsample_mean=float(np.mean([r['ARI'] for r in boot if r['kind']=='subsample'])),
                 stability_seed_mean=float(np.mean([r['ARI'] for r in boot if r['kind']=='seed'])),external=ext,
                 runtime_seconds=round(time.time()-start,2))
    dump(out/'summary.json',summary);dump(out/'config_used.json',cfg)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config.yaml');args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text());fit(cfg)
