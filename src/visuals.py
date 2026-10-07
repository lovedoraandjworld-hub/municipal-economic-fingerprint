"""Publication figures and self-contained interactive evidence browser."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from plotly.offline import get_plotlyjs
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results';F=R/'figures';F.mkdir(exist_ok=True)
COLORS=['#157F6B','#3D64B1','#D78F38','#AB4E71','#8665B4','#367B94','#788039','#B76441','#778091','#AD6D94']
NAMES={1:'Транспорт и общепит',5:'Умеренная доля маркетплейсов',9:'Маркетплейсы и услуги',11:'Продовольствие и маркетплейсы',13:'Общепит и здоровье',14:'Продовольственный смешанный',15:'Диверсифицированный сервисный'}
def main():
 s=json.loads((R/'summary.json').read_text());p=np.load(R/'panel.npz');dates=p['dates'].tolist();ids=p['ids'].tolist();labels=p['labels'];shares=p['shares'];raw=p['raw_shares'];cats=p['categories'].tolist()
 profiles=pd.read_csv(R/'profiles.csv');latest=profiles[profiles.date==dates[-1]].sort_values('Продовольствие',ascending=False)
 metrics=pd.read_csv(R/'metrics.csv');stable=pd.read_csv(R/'temporal_stability.csv');conf=pd.read_csv(R/'confidence_latest.csv')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold'})
 fig,ax=plt.subplots(figsize=(11,6));left=np.zeros(len(latest))
 for j,c in enumerate(cats):
  vals=latest[c].values*100;ax.barh(np.arange(len(latest)),vals,left=left,label=c,color=COLORS[j]);left+=vals
 ax.set_yticks(np.arange(len(latest)),[f'G{int(c)} · {int(n)} МО' for c,n in zip(latest.cluster,latest['size'])]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Средняя доля расходов в группе, %');ax.set_title('Профили групп · июль–декабрь 2024');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,frameon=False);fig.tight_layout();fig.savefig(F/'profiles.png',dpi=180,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,5));last=metrics[metrics.date==dates[-1]]
 for ax,metric,title in zip(axes,['SW','AVI'],['Разделение профилей: SW ↑','Изоляция групп в сети: AVI ↑']):
  ax.bar(last.method,last[metric],color=COLORS[:3]);ax.set_ylim(0,max(last[metric])*1.2);ax.set_title(title)
  for i,v in enumerate(last[metric]):ax.text(i,v+.015,f'{v:.3f}',ha='center')
 fig.suptitle('Одинаковое число групп K = 7, одинаковые данные');fig.tight_layout();fig.savefig(F/'comparison.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,4));ax.plot(stable.date,stable.ARI,'o-',color=COLORS[0],label='ARI соседних месяцев');ax.plot(stable.date,stable.switch_share,'o-',color=COLORS[2],label='Доля переходов после сопоставления');ax.set_ylim(0,1);ax.set_ylabel('Значение');ax.set_xticks(range(0,len(stable),3),stable.date.iloc[::3],rotation=35);ax.legend(frameon=False);ax.set_title('Динамика групп после сглаживания за шесть месяцев');fig.tight_layout();fig.savefig(F/'stability.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,6));xy=p['coords'][-1];edge=pd.read_csv(R/'edges_latest.csv').sort_values('weight',ascending=False).head(700);index={v:i for i,v in enumerate(ids)}
 segments=[[xy[index[a]],xy[index[b]]] for a,b in zip(edge.source,edge.target)];ax.add_collection(LineCollection(segments,color='#d8dce2',linewidths=.4,zorder=0))
 groups=sorted(np.unique(labels[-1]).tolist());cluster_colors={int(c):COLORS[j%len(COLORS)] for j,c in enumerate(groups)}
 for c in groups:
  mask=labels[-1]==c;ax.scatter(xy[mask,0],xy[mask,1],s=9,alpha=.65,color=cluster_colors[int(c)],label=f'G{c}')
 ax.set_xlabel('PC1, координата профиля');ax.set_ylabel('PC2, координата профиля');ax.set_title('Экономическая близость · декабрь 2024');ax.legend(ncol=4,frameon=False);fig.tight_layout();fig.savefig(F/'network.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,4));m=pd.DataFrame(s['external']['market_access']);ax.bar(m.cluster.astype(str),m['median'],color=[cluster_colors[int(c)] for c in m.cluster]);ax.errorbar(range(len(m)),m['median'],yerr=[m['median']-m.q25,m.q75-m['median']],fmt='none',color='#263142',capsize=5);ax.set_ylabel('Индекс доступности рынков');ax.set_xlabel('Группа');ax.set_title('Внешняя проверка: медиана и межквартильный диапазон');fig.tight_layout();fig.savefig(F/'market.png',dpi=180);plt.close(fig)
 change=raw[12:,:,2]-raw[:12,:,2]
 diagnostics={'marketplace_yoy_mean_pp':float(change.mean()*100),'marketplace_yoy_median_pp':float(np.median(change)*100),
              'marketplace_yoy_positive_fraction':float((change>0).mean()),'marketplace_2023_mean_pct':float(raw[:12,:,2].mean()*100),
              'marketplace_2024_mean_pct':float(raw[12:,:,2].mean()*100),'food_2023_mean_pct':float(raw[:12,:,0].mean()*100),
              'food_2024_mean_pct':float(raw[12:,:,0].mean()*100),'core_count':int((conf.seed_support>=.8).sum()),'boundary_count':int((conf.seed_support<.8).sum()),
              'names':{str(c):(NAMES.get(c,'Профиль '+str(c)) if s['neighbors']==12 and s['smoothing_months']==6 and s['selected_resolution']==.5 and json.loads((R/'config_used.json').read_text())['seed']==42 else 'Профиль '+str(c)) for c in groups}}
 (R/'interpretation.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2))
 pack={'cluster_colors':cluster_colors,'summary':s,'interpretation':diagnostics,'dates':dates,'ids':ids,'categories':cats,'labels':labels.tolist(),
       'coords':np.round(p['coords'],5).tolist(),'shares':np.round(shares,5).tolist(),'raw':np.round(raw,5).tolist(),
       'totals':p['totals'].tolist(),'confidence':conf.seed_support.round(4).tolist(),
       'metrics':json.loads(metrics.to_json(orient='records')),'stability':json.loads(stable.to_json(orient='records'))}
 template=(ROOT/'src/dashboard_template.html').read_text()
 html=template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(pack,ensure_ascii=False,separators=(',',':')))
 (ROOT/'docs/interactive.html').write_text(html)
 print(json.dumps(diagnostics,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
