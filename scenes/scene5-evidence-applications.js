/* Local application revision. Supersedes scene 07's decorative graph drawing,
   while preserving the scene identity and existing scroll/rail orchestration.
   Numerical evidence is loaded from a frozen, provenance-bound review export. */
(function () {
  'use strict';
  window.EBScenes = window.EBScenes || {};
  window.EBScenes.platform = function(p, ctx) {
    p.setup = function(){p.createCanvas(ctx.W,ctx.H);p.clear();p.noLoop();};
    p.windowResized = function(){p.resizeCanvas(ctx.W,ctx.H);p.clear();};
  };
  const dialog=document.getElementById('applicationDialog');
  const host=document.getElementById('applicationCase');
  const tabs=document.querySelector('.application-case-tabs');
  const open=document.getElementById('applicationOpen');
  let payload=null, selected='locate', handoff=false;
  function node(tag,text,klass){const e=document.createElement(tag);e.textContent=text;if(klass)e.className=klass;return e;}
  function render(){
    if(!payload)return;
    const c=payload.cases.find(r=>r.id===selected);
    tabs.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.case===selected)));
    host.replaceChildren(node('p',c.setting+' / '+c.record,'application-case-heading'),node('h3',c.question),node('p',c.finding));
    for(const [title,body] of [['Decision now',c.decision],['Next action',c.next_action]])host.append(node('h4',title),node('p',body));
    host.append(node('p',c.numbers,'application-evidence-number'),node('p',c.boundary,'application-case-boundary'));
    if(c.id==='locate'){
      const wrap=node('div','','application-lookup'),table=document.createElement('table'),head=document.createElement('thead'),tr=document.createElement('tr');
      for(const t of ['Source depth','Ammonium mg/kg','pH','Salinity'])tr.append(node('th',t));head.append(tr);table.append(head);
      const body=document.createElement('tbody');
      for(const r of payload.depths){const row=document.createElement('tr'),th=node('th',r.depth);th.scope='row';row.append(th,node('td',r.ammonium_mg_kg.toFixed(2)),node('td',r.ph.toFixed(2)),node('td',r.salinity_psu===null?'Missing':String(r.salinity_psu)));body.append(row);}
      table.append(body);wrap.append(table);host.append(wrap);
    }
    const sources=node('details','','application-dialog-note application-sources');
    sources.append(node('summary','Evidence sources and attribution'),node('p','Snapshot '+payload.snapshot+'. '+payload.review_method));
    for(const source of (payload.public_sources||[]).filter(s=>s.id===c.source_id)){
      const citation=node('p',source.creators+' '),link=node('a',source.title+' · version '+source.version);
      link.href=source.url;link.target='_blank';link.rel='noopener';citation.append(link);
      const license=node('a',source.license);license.href=source.license_url;license.target='_blank';license.rel='noopener';
      citation.append(document.createTextNode(' · '),license);sources.append(citation,node('p',source.modifications));
    }
    host.append(sources);
    const explore=node('a','Follow this case in the evidence network','application-network-link');
    explore.href='#scene-network';
    // Hand keyboard focus to the matching explorer case instead of returning it
    // to this scene's opener, which the handoff scrolls out of view.
    explore.addEventListener('click',event=>{event.preventDefault();handoff=true;dialog.close();document.dispatchEvent(new CustomEvent('emergentbiome:network-case',{detail:{caseId:c.id,focus:true}}));});
    host.append(explore);
  }
  async function load(){
    if(payload){render();return;}
    try{
      const response=await fetch('data/molecular-application-cases-public-v1.json');if(!response.ok)throw new Error('Evidence file could not be loaded.');
      payload=await response.json();
      if(!Array.isArray(payload.cases)||payload.cases.length!==3||!Array.isArray(payload.depths)||payload.depths.length!==5)throw new Error('Evidence contract is incomplete.');
      tabs.replaceChildren();
      for(const c of payload.cases){const b=node('button',c.id==='interpret'?'Interpret the gene':c.id==='locate'?'Resolve the depth':'Choose the next test');b.type='button';b.dataset.case=c.id;b.addEventListener('click',()=>{selected=c.id;render();});tabs.append(b);}
      render();
    }catch(error){host.replaceChildren(node('p','The evidence file is temporarily unavailable. Please retry to load the recorded cases.'),node('p',error.message));const retry=node('button','Retry evidence load');retry.type='button';retry.addEventListener('click',()=>{payload=null;load();});host.append(retry);}
  }
  open.addEventListener('click',()=>{dialog.showModal();load();});
  document.getElementById('applicationClose').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('close',()=>{if(handoff){handoff=false;return;}open.focus({preventScroll:true});});
})();
