"""Internal validity in attribute and graph spaces. Explicit variants, no hidden aliases."""
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def s_dbw_geometric(X, labels):
    """Halkidi density-within-radius variant; variance norms, population variance.

    r=sqrt(sum_k ||var(C_k)||)/K. Density counts points INSIDE this radius.
    Between density is evaluated on C_k union C_l; zero denominator => NaN.
    """
    groups = [X[labels == c] for c in np.unique(labels)]
    K = len(groups)
    if K < 2 or K >= len(X): return float('nan')
    centers = [g.mean(0) for g in groups]
    vn = np.array([np.linalg.norm(g.var(0)) for g in groups])
    norm = np.linalg.norm(X.var(0))
    if norm == 0: return float('nan')
    radius = np.sqrt(vn.sum()) / K
    density = [np.count_nonzero(np.linalg.norm(g - c, axis=1) <= radius) for g,c in zip(groups,centers)]
    inter = []
    for i in range(K):
        for j in range(i+1,K):
            denominator = max(density[i],density[j])
            if denominator == 0: return float('nan')
            mid = (centers[i]+centers[j])/2
            union = np.vstack([groups[i],groups[j]])
            inter.append(np.count_nonzero(np.linalg.norm(union-mid,axis=1)<=radius)/denominator)
    return float(vn.mean()/norm + np.mean(inter))

def s_dbw(X, labels):
    """Pinned s-dbw 0.4.0, Halkidi variant, nearest observed centroid.

    Snapping the density center to an observation prevents empty-center
    denominators for thin manifolds. This explicit convention differs from
    the geometric-centroid diagnostic retained above.
    """
    from s_dbw import S_Dbw
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(X): return float('nan')
    if np.linalg.norm(X.var(0)) == 0: return float('nan')
    return float(S_Dbw(X, labels, method='Halkidi', nearest_centr=True, metric='euclidean'))

def graph_scores(graph, labels):
    """AVI/AVU eq.18-21 Shalileh et al. DOI 10.1134/S1064562425700589.

    Undirected, weighted graph, no self loops. MQ is density-based classic
    variant: mean intra density minus mean inter density, not modularity Q.
    Zero exterior degree contributes 0 to undefined pair ratio.
    """
    labs, y = np.unique(labels, return_inverse=True)
    K=len(labs); sizes=np.bincount(y,minlength=K)
    block=np.zeros((K,K))
    for (i,j),w in zip(graph.get_edgelist(),graph.es['weight']):
        block[y[i],y[j]]+=w; block[y[j],y[i]]+=w
    internal=np.diag(block); external=block.sum(1)-internal
    avi=float(np.mean(np.divide(internal,internal+external,out=np.zeros(K),where=internal+external>0)))
    pair=[]; inter_density=[]
    for k in range(K):
        for l in range(k+1,K):
            denom=external[k]+external[l]-block[k,l]
            pair.append(block[k,l]/denom if denom>0 else 0.)
            inter_density.append(block[k,l]/(sizes[k]*sizes[l]))
    avu=float(2*sum(pair)/K) if K else float('nan')
    intra=np.divide(internal,sizes*(sizes-1),out=np.zeros(K),where=sizes>1)
    mq=float(intra.mean()-(np.mean(inter_density) if inter_density else 0))
    q=float(graph.modularity(y.tolist(),weights='weight')) if graph.ecount() else float('nan')
    return dict(AVI=avi,AVU=avu,MQ=mq,Q=q)

def evaluate(X, labels, graph, sample=1000, seed=42):
    K=len(np.unique(labels)); n=len(labels)
    result=dict(K=K,N=n,min_size=int(np.bincount(np.unique(labels,return_inverse=True)[1]).min()))
    result.update(graph_scores(graph,labels))
    if 1<K<n:
        result.update(SW=float(silhouette_score(X,labels,sample_size=min(sample,n),random_state=seed)),
                      CH=float(calinski_harabasz_score(X,labels)), S_Dbw=s_dbw(X,labels))
        result['CH_per_N']=result['CH']/n
    else:result.update(SW=float('nan'),CH=float('nan'),S_Dbw=float('nan'),CH_per_N=float('nan'))
    return result
