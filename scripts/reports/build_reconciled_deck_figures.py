#!/usr/bin/env python3
"""Publication figures from the reconciled atlas, with machine-readable provenance."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from atlas_embedding_contract import validate_embedding_contract

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'results/reports/emergentbiome_molecular_atlas_20260929_layer33'
OUT=ROOT/'results/reports/ecosphere_grant_deck_20260929'
RUN=ROOT/'results/blue_catalyst_poc/reembed_single_configuration_20260928'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    conf=json.loads((REPORT/'embedding_configuration.json').read_text())
    contract_path=ROOT/'configs/atlas_embedding_contract_20260929.json'
    contract=json.loads(contract_path.read_text())
    verified=validate_embedding_contract(ROOT,contract_path,
        [{'path':(ROOT/r['path']).parent} for r in contract['artifacts']])
    assert verified['contract_sha256']==conf['contract_sha256']
    p=REPORT/'assets/data/niche.json'
    assert conf['status']=='verified' and sha(p)==conf['niche_sha256']
    data=json.loads(p.read_text()); nodes=[n for n in data['nodes'] if n.get('umap_1') is not None]
    assert len(nodes)==7710
    groups=[('rumen__','Rumen reference','#E08A4A'),('mucc','Wetland','#9ECEA5'),
            ('msm_china_2025__','Mangrove · China coast','#65C8DA'),('futian_mangrove_2026_qi__','Mangrove · Futian','#AFA8EE')]
    assert all(sum(n['proteome_id'].startswith(pre) for pre,_,_ in groups)==1 for n in nodes)
    plt.rcParams.update({'font.family':'Liberation Sans','text.color':'#E7E4DC','font.size':10})
    fig,ax=plt.subplots(figsize=(11,4.9),facecolor='#0b4243');ax.set_facecolor('#0b4243')
    for pre,label,col in groups:
        ns=[n for n in nodes if n['proteome_id'].startswith(pre)]
        ax.scatter([n['umap_1'] for n in ns],[n['umap_2'] for n in ns],s=3,alpha=.7,c=col,label=f'{label} ({len(ns):,})',linewidths=0,rasterized=True)
    ax.set_axis_off();ax.legend(loc='upper right',frameon=False,markerscale=2.5,fontsize=12)
    ax.text(.99,.02,'UMAP · final-layer ESM-2\nExploratory geometry; axes have no physical units',transform=ax.transAxes,ha='right',va='bottom',color='#AEC4C2',fontsize=11)
    fig.subplots_adjust(left=.01,right=.99,top=.98,bottom=.02)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'assets/atlas_reconciled.{ext}',dpi=240,facecolor=fig.get_facecolor())
    plt.close(fig)
    # Same-DNA positive controls: pilot queries against all 2,501 MUCC targets.
    pilot=np.load(ROOT/'results/blue_catalyst_poc/reconciled_layer33_20260929/artifacts/genome_embeddings.npz')
    idx={s:i for i,s in enumerate(pilot['proteome_id'].astype(str))}
    rows=list(csv.DictReader((RUN/'configuration_evidence/matched_configuration_retrieval.tsv').open(),delimiter='\t'))
    ids=[];vec=[]
    for r in csv.DictReader((ROOT/'configs/methanet_atlas_lanes_20260929.tsv').open(),delimiter='\t'):
        if r['lane_id']=='mucc_v1_owc_wetland':
            for folder in r['esm2_artifacts_dirs'].split(';'):
                z=np.load(ROOT/folder/'genome_embeddings.npz',allow_pickle=True)
                ids.extend(z['proteome_id'].astype(str));vec.extend(z['embeddings'])
    vv=np.asarray(vec,dtype=np.float64);vv/=np.linalg.norm(vv,axis=1,keepdims=True)
    assert len(ids)==2501 and len(set(ids))==2501
    old=np.load(ROOT/'results/blue_catalyst_poc/runs/apolo_full_20260228_080644_embed_20260305_061952/artifacts/genome_embeddings.npz',allow_pickle=True)
    oldidx={s:i for i,s in enumerate(old['sample'].astype(str))}
    controls=[]
    for r in rows:
        q=pilot['embeddings'][idx[r['poc_proteome_id']]].astype(np.float64);q/=np.linalg.norm(q)
        sim=vv@q;j=ids.index(r['mucc_proteome_id']);rank=1+int((sim>sim[j]+1e-12).sum())
        oq=old['embeddings'][oldidx[r['poc_proteome_id']]].astype(np.float64);oq/=np.linalg.norm(oq)
        osim=vv@oq;oldrank=1+int((osim>osim[j]+1e-12).sum())
        assert oldrank==int(r['archived_rank'])
        controls.append({'pilot_id':r['poc_proteome_id'],'target_id':r['mucc_proteome_id'],
                         'archived_rank':oldrank,'reconciled_rank':rank,'cosine':float(sim[j])})
    assert len(controls)==23 and all(r['reconciled_rank']==1 for r in controls)
    with (OUT/'same_dna_controls.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(controls[0]),delimiter='\t');w.writeheader();w.writerows(controls)
    vr=json.loads((RUN/'verify_and_collate_summary.json').read_text())
    assert vr['status']=='pass' and vr['records_ok']==662
    gate=vr['gates']['pilot_reproduction']
    result={'headline':'662 / 662 verified · 23 / 23 same-DNA controls recovered first',
            'detail':'All pilot records passed retained-input and reproduction checks. Each same-DNA control is the closest match among 2,501 MUCC targets. This establishes configuration consistency for the controls; ecological prediction still needs independent tests.',
            'pilot_records':662,'same_dna_controls':23,'target_records':2501,'pilot_reproduction_gate':gate,
            'mean_reconciled_control_cosine':float(np.mean([r['cosine'] for r in controls])),
            'median_archived_rank':float(np.median([r['archived_rank'] for r in controls])),
            'minimum_reconciled_control_cosine':min(r['cosine'] for r in controls)}
    (OUT/'reconciliation_slide.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(9,3.8),facecolor='#0b4243');ax.set_facecolor('#0b4243')
    ordered=sorted(controls,key=lambda r:r['archived_rank']);x=np.arange(1,24)
    before=[r['archived_rank'] for r in ordered];after=[r['reconciled_rank'] for r in ordered]
    ax.vlines(x,after,before,color='#437672',lw=1,zorder=1)
    ax.scatter(x,before,color='#E08A4A',s=32,label='Mixed pooling (superseded)',zorder=3)
    ax.scatter(x,after,color='#78C6BB',s=32,label='Reconciled layer 33',zorder=3)
    ax.set_yscale('log');ax.set_ylim(1600,.55);ax.set_yticks([1,10,100,1000],labels=['1','10','100','1,000'])
    ax.set_xticks([1,6,12,18,23]);ax.set_xlabel('23 selected same-DNA controls, sorted by old rank',color='#AEC4C2')
    ax.set_ylabel('Match rank among 2,501 targets\n(smaller is better)',color='#AEC4C2')
    ax.tick_params(colors='#AEC4C2');ax.minorticks_off()
    for sp in ax.spines.values():sp.set_visible(False)
    ax.grid(axis='y',alpha=.15)
    ax.legend(loc='center right',frameon=True,facecolor='#0b4243',edgecolor='#0b4243',framealpha=1,fontsize=10)
    fig.subplots_adjust(left=.13,right=.99,top=.96,bottom=.18)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'assets/configuration_controls.{ext}',dpi=240,facecolor=fig.get_facecolor())
    plt.close(fig)
    provenance={'geometry_configuration':conf,'source_niche':str(p.relative_to(ROOT)),
                'source_sha256':sha(p),'plotted_records':len(nodes),'selection':'all records with computed UMAP; no subsampling',
                'claim_boundary':'Exploratory molecular geometry, not functional transfer, flux prediction or MRV validation',
                'controls':result,'figures':[{ 'path':f'assets/{name}.{ext}','sha256':sha(OUT/f'assets/{name}.{ext}')} for name in ['atlas_reconciled','configuration_controls'] for ext in ['png','pdf','svg']]}
    (OUT/'figure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
