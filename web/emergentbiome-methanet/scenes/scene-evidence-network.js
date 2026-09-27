/* A bounded view over three frozen evidence cases. The presentation graph and
   exact canonical source assertions are distinct, inspectable layers. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const section = $('scene-network');
  if (!section) return;
  const shell = $('networkShell'), canvas = $('networkCanvas'), layer = $('networkGraphLayer');
  const dialog = $('networkImmersiveDialog');
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const labels = {recorded:'Recorded evidence',review:'Review required',pending:'Unresolved',action:'Scientific recommendation'};
  const caseTitles = {interpret:'Interpret the gene',locate:'Resolve the depth',design:'Choose the next test'};
  const state = {data:null,caseId:'interpret',focus:null,selected:null,source:null,sourceSelection:null,reviewPage:0,page:0,view:'graph',zoom:1,pan:{x:0,y:0},loading:false};
  let activeNodes = [], activeEdges = [], resizeFrame = null, priorFocus = null;
  function el(tag, text, cls) { const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n; }
  function button(text, action, cls) {const b=el('button',text,cls);b.type='button';b.addEventListener('click',action);return b;}
  function current() {return state.data.cases.find(c=>c.id===state.caseId);}
  function getNode(id) {return current().nodes.find(n=>n.id===id);}
  function parent(id) {return current().edges.find(e=>e.target===id)?.source;}
  function announce(text) {$('networkLive').textContent=text;}
  function focusRecord(id, preventScroll=true) {
    const selector=state.view==='list'?'[data-record-id]':'[data-node-id]';
    [...shell.querySelectorAll(selector)].find(b=>(b.dataset.recordId||b.dataset.nodeId)===id)?.focus({preventScroll});
  }
  function fit() {state.zoom=1;state.pan={x:0,y:0};transform();}
  function transform() {layer.style.transform=`translate(${state.pan.x}px,${state.pan.y}px) scale(${state.zoom})`;}
  function status(s) {const n=el('span',labels[s]||s,'net-status');n.dataset.state=s;return n;}
  function facts(rows) {const dl=el('dl',undefined,'net-facts');for(const [key,value] of rows){const d=el('div');d.append(el('dt',key),el('dd',value));dl.append(d);}return dl;}
  function shortTerm(value) {
    if(value.startsWith('<'))return decodeURIComponent(value.slice(1,-1).split(/[\/#]/).pop());
    const match=value.match(/^"((?:\\.|[^"\\])*)"/);
    if(match){try{return JSON.parse('"'+match[1]+'"');}catch(e){return match[1];}}
    return value;
  }
  function sourceModel() {
    const selected=getNode(state.selected)||getNode(current().root);
    const entries=state.data.canonical_assertions[state.source];
    const offset=state.page*6;
    const record={id:'source:record',label:'Source record',kind:'source',state:'recorded',
      subtitle:'Selected public facts',summary:'A source record with an explicitly selected set of public facts.',
      facts:[['Canonical subject',state.source]],children:[],source_refs:[state.source],queries:selected.queries};
    const origin={...selected,id:'source:origin',children:[]};
    const nodes=[origin,record], edges=[{id:'source-origin',source:origin.id,target:record.id,
      label:'inspect source evidence',kind:'navigation',meaning:'View link to a source evidence object.'}];
    entries.slice(offset,offset+6).forEach((r,index)=>{
      const predicate=r.predicate.split(/[\/#]/).pop();
      const fullLabel=String(shortTerm(r.object));
      const node={id:'assertion:'+(offset+index),label:fullLabel.length>48?fullLabel.slice(0,45)+'…':fullLabel,
        fullLabel,kind:'assertion',state:'recorded',subtitle:predicate,summary:'An exact, allowlisted predicate and object retained from the frozen source projection.',
        facts:[['Subject',state.source],['Predicate',r.predicate],['Object',r.object]],children:[],
        source_refs:[state.source],queries:selected.queries};
      nodes.push(node);record.children.push(node.id);
      edges.push({id:'canonical:'+(offset+index),source:record.id,target:node.id,label:predicate,kind:'canonical',
        meaning:'Selected exact source fact. The full internal evidence packet is outside the public view.',facts:node.facts});
    });
    return {nodes,edges,root:origin.id,focus:record.id,total:entries.length};
  }
  function reviewModel() {
    const c=current(), focus=getNode(state.focus)||getNode(c.root);
    const visible=focus.children.slice(state.page*6,state.page*6+6);
    const ids=focus.id===c.root?[c.root,...visible]:[c.root,focus.id,...visible];
    const nodes=[...new Set(ids)].map(getNode);
    const edges=nodes.filter(n=>n.id!==c.root&&n.id!==focus.id).map(n=>c.edges.find(e=>e.target===n.id));
    if(focus.id!==c.root){
      const direct=c.edges.find(e=>e.source===c.root&&e.target===focus.id);
      edges.unshift(direct||{id:'path:'+focus.id,source:c.root,target:focus.id,label:'follow evidence path',kind:'navigation',
        meaning:'Condensed navigation path. The breadcrumb preserves each intermediate record.'});
    }
    return {nodes,edges,root:c.root,focus:focus.id,total:focus.children.length};
  }
  function positions(model,width,height) {
    const out=new Map(), narrow=width<480;
    if(model.root===model.focus){
      out.set(model.root,{x:width*.5,y:height*.49});
      const corners=narrow?[[.23,.18],[.77,.18],[.23,.79],[.77,.79]]:[[.23,.22],[.78,.22],[.23,.77],[.78,.77]];
      model.nodes.filter(n=>n.id!==model.root).forEach((n,i)=>out.set(n.id,{x:width*corners[i][0],y:height*corners[i][1]}));
    }else{
      out.set(model.root,{x:width*(narrow?.25:.16),y:height*(narrow?.14:.5)});
      out.set(model.focus,{x:width*(narrow?.75:.46),y:height*(narrow?.14:.5)});
      const leaves=model.nodes.filter(n=>n.id!==model.root&&n.id!==model.focus);
      leaves.forEach((n,i)=>{
        if(narrow){const rows=Math.ceil(leaves.length/2);out.set(n.id,{x:width*(i%2?.75:.25),y:rows===1?260:210+Math.floor(i/2)*(height-290)/Math.max(1,rows-1)});}
        else out.set(n.id,{x:width*.8,y:60+(leaves.length===1?(height-120)/2:i*(height-120)/(leaves.length-1))});
      });
    }
    return out;
  }
  function pathFor(id) {
    const path=[];let at=getNode(id);
    while(at){path.unshift(at);at=getNode(parent(at.id));}
    return path;
  }
  function breadcrumbs() {
    const nav=$('networkBreadcrumb');nav.replaceChildren();
    pathFor(state.focus).forEach((n,i)=>{
      if(i)nav.append(el('span','/'));
      nav.append(button(i===0?'Review profile':n.label,()=>select(n.id)));
    });
    if(state.source){nav.append(el('span','/'));nav.append(button('Selected source facts',()=>sourceInspect(state.source)));}
  }
  function chooseCase(id) {
    if(!state.data){state.caseId=id;load();return;}
    if(!state.data.cases.some(c=>c.id===id))return;
    state.caseId=id;state.focus=current().root;state.selected=current().root;state.source=null;state.sourceSelection=null;state.reviewPage=0;state.page=0;
    $('networkSearch').value='';$('networkSearchResults').hidden=true;
    fit();render();announce(caseTitles[id]+'. '+current().status+'.');
  }
  function select(id) {
    const n=getNode(id);if(!n)return;
    state.source=null;state.selected=id;state.page=0;
    state.focus=n.children.length?n.id:(parent(n.id)||current().root);
    if(!n.children.length)state.page=Math.floor(getNode(state.focus).children.indexOf(n.id)/6);
    $('networkSearch').value='';$('networkSearchResults').hidden=true;
    fit();render();announce(n.label+'. '+labels[n.state]+'. '+n.children.length+' related records.');
    focusRecord(id);
  }
  function sourceInspect(ref) {
    if(!state.data.canonical_assertions[ref])return;
    if(!state.source)state.reviewPage=state.page;
    state.source=ref;state.sourceSelection='source:record';state.page=0;fit();render();
    focusRecord('source:record',false);
    announce('Selected source facts. '+state.data.canonical_assertions[ref].length+' facts available.');
  }
  function returnToReview() {
    state.source=null;state.sourceSelection=null;state.page=state.reviewPage;fit();render();
    focusRecord(state.selected,false);
    announce('Returned to the selected evidence and its review page.');
  }
  function selectSourceNode(n) {
    if(n.id==='source:origin'){returnToReview();return;}
    state.sourceSelection=n.id;renderDetails(n);
    shell.querySelectorAll('[data-node-id],[data-record-id]').forEach(b=>b.setAttribute('aria-pressed',String((b.dataset.nodeId||b.dataset.recordId)===n.id)));
    announce(n.label+'. Selected source fact.');
  }
  function render() {
    const c=current();
    $('networkCases').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.networkCase===c.id)));
    const profile=$('networkProfile');profile.replaceChildren();
    for(const row of c.profile){
      const b=button('',()=>select(c.id+':'+row.key));b.dataset.profileKey=row.key;b.dataset.state=row.state;
      b.setAttribute('aria-label',row.label+': '+row.status+'. Explore related evidence.');
      b.append(el('span',row.label),el('b',row.status));profile.append(b);
    }
    breadcrumbs();renderGraph();renderList();
    const detailNode=state.source?sourceModel().nodes.find(n=>n.id===state.sourceSelection)||sourceModel().nodes[1]:getNode(state.selected)||getNode(c.root);
    renderDetails(detailNode);
    const model=state.source?sourceModel():reviewModel();
    $('networkPagination').hidden=model.total<=6;
    $('networkPageStatus').textContent=`${state.page*6+1} to ${Math.min(model.total,state.page*6+6)} of ${model.total} ${state.source?'source facts':'related records'}`;
    $('networkPagePrevious').disabled=state.page===0;
    $('networkPageNext').disabled=(state.page+1)*6>=model.total;
    $('networkGraphMode').textContent=state.source?'SELECTED SOURCE FACTS':'REVIEW PATHS';
    $('networkMapHint').textContent=state.source?'Select a linked object to inspect its source fact.':'Select a node to unfold its evidence.';
  }
  function renderGraph() {
    if(!state.data||state.view==='list')return;
    // A passive resize may replace the focused mark. Restore only focus that
    // was already inside this graph, never focus from another page control.
    const focused=layer.contains(document.activeElement)?document.activeElement:null;
    const focusedNode=focused?.dataset.nodeId,focusedEdge=focused?.dataset.edgeId;
    const model=state.source?sourceModel():reviewModel();
    const width=canvas.clientWidth||700,narrow=width<480;
    const reading=document.documentElement.classList.contains('eb-text-enlarged');
    const leafCount=model.nodes.length-2;
    if(!reading)canvas.style.height=(narrow?Math.max(540,Math.ceil(leafCount/2)*105+230):Math.max(460,leafCount*84+100))+'px';
    let height=canvas.clientHeight;
    const coords=positions(model,width,height);
    activeNodes=model.nodes;activeEdges=model.edges;
    const host=$('networkNodes');host.replaceChildren();
    model.nodes.forEach(n=>{
      const point=coords.get(n.id),isRoot=n.id===model.root,isFocus=n.id===model.focus;
      let cls='net-node'+(isRoot?' net-node--profile':isFocus?' net-node--focus':'');
      if(!isRoot&&!isFocus&&model.root!==model.focus)cls+=' net-node--leaf';
      const b=button('',()=>{
        if(state.source)selectSourceNode(n);
        else select(n.id);
      },cls);
      b.dataset.nodeId=n.id;b.dataset.state=n.state;
      b.style.left=point.x+'px';b.style.top=point.y+'px';
      b.setAttribute('aria-pressed',String(n.id===(state.source?state.sourceSelection:state.selected)));
      b.setAttribute('aria-label',(n.fullLabel||n.label)+'. '+labels[n.state]+(n.children.length?'. Expand '+n.children.length+' related records.':'. Inspect evidence.'));
      b.append(el('strong',n.label),el('small',isRoot&&!state.source?'REVIEW IN PROGRESS':n.subtitle));
      if(n.children.length&&!isRoot)b.append(el('span',String(n.children.length),'net-node-count'));
      host.append(b);
    });
    // Enlarged text uses measured cards in a reading column. Geometry remains
    // navigational; every source relationship and inspector action is retained.
    if(reading){
      let top=70;
      for(const b of host.children){
        const h=b.offsetHeight;
        const point={x:width/2,y:top+h/2};coords.set(b.dataset.nodeId,point);
        b.style.left=point.x+'px';b.style.top=point.y+'px';top+=h+48;
      }
      canvas.style.height=Math.max(460,top+90)+'px';height=canvas.clientHeight;
    }
    const svg=$('networkEdges');svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
    for(const edge of model.edges){
      const a=coords.get(edge.source),b=coords.get(edge.target);if(!a||!b)continue;
      const path=reading?`M ${a.x} ${a.y} C 14 ${a.y}, 14 ${b.y}, ${b.x} ${b.y}`:
        `M ${a.x} ${a.y} C ${(a.x+b.x)/2} ${a.y}, ${(a.x+b.x)/2} ${b.y}, ${b.x} ${b.y}`;
      for(const hit of [false,true]){
        const mark=document.createElementNS('http://www.w3.org/2000/svg','path');mark.setAttribute('d',path);
        mark.setAttribute('class',hit?'net-edge-hit':'net-edge');mark.dataset.kind=edge.kind;
        if(hit){mark.dataset.edgeId=edge.id;mark.setAttribute('role','button');mark.setAttribute('tabindex','0');mark.setAttribute('aria-label','Inspect relationship: '+edge.label);
          mark.addEventListener('click',()=>renderEdge(edge));mark.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();renderEdge(edge);}});
        }else mark.setAttribute('aria-hidden','true');
        svg.append(mark);
      }
    }
    $('networkVisibleCount').textContent=model.nodes.length+' visible records';
    transform();
    if(focusedNode)focusRecord(focusedNode,false);
    else if(focusedEdge)[...svg.querySelectorAll('[data-edge-id]')].find(n=>n.dataset.edgeId===focusedEdge)?.focus();
  }
  function changePage(delta) {
    const origin=document.activeElement.id;
    state.page+=delta;
    if(state.source)state.sourceSelection='source:record';
    fit();render();
    const control=$(origin);
    if(control&&!control.disabled)control.focus({preventScroll:true});
    else focusRecord(state.source?'source:record':state.focus);
    announce($('networkPageStatus').textContent+'.');
  }
  function renderEdge(edge) {
    const data={label:edge.label,kind:'relationship',state:edge.kind==='canonical'?'recorded':'review',
      summary:edge.meaning||'This relationship organizes the selected case evidence. Inspect the linked source assertions for exact ontology predicates.',
      facts:edge.facts||[['Relationship type',edge.kind],['From',edge.source],['To',edge.target]],
      source_refs:edge.source_refs||[],queries:[],children:[]};
    renderDetails(data);announce('Relationship: '+edge.label);
  }
  function renderDetails(n) {
    const detail=$('networkDetail'),c=current();detail.replaceChildren();
    detail.append(el('p',c.setting,'net-eyebrow'),status(n.state),el('h3',n.kind==='profile'?c.question:(n.fullLabel||n.label)),el('p',n.summary));
    if(n.kind==='profile'){
      detail.append(el('h4','Decision now'),el('p',c.decision),el('h4','Next observation'),el('p',c.next_action,'net-next'));
      detail.append(button('Why is this interpretation on hold?',()=>select(c.id+':review'),'net-inspect-action'));
    }
    const profileRow=c.profile.find(r=>r.key===n.key);
    if(profileRow?.basis)detail.append(el('h4','Basis for this status'),el('p',profileRow.basis));
    if(n.facts.length)detail.append(facts(n.facts));
    if(n.children.length&&n.kind!=='profile'){
      detail.append(el('h4','Follow the evidence'));
      const ids=n.id===state.focus?n.children.slice(state.page*6,state.page*6+6):n.children.slice(0,6);
      for(const id of ids){const child=getNode(id);if(child){const b=button(child.label,()=>select(id),'net-inspect-action');b.append(el('span','↗'));detail.append(b);}}
      if(n.children.length>6)detail.append(el('p',`${n.children.length} related records are available. Use the network page controls or search to inspect them.`));
    }
    if(state.source){
      const entries=state.data.canonical_assertions[state.source],count=Math.ceil(entries.length/6);
      detail.append(el('h4','Selected source facts'),el('p',`Showing facts ${state.page*6+1} to ${Math.min(entries.length,(state.page+1)*6)} of ${entries.length}.`));
      const pages=el('div',undefined,'net-source-pages');
      const prev=button('Previous source facts',()=>changePage(-1),'net-inspect-action');prev.id='networkSourcePrevious';prev.disabled=state.page===0;
      const next=button('More source facts',()=>changePage(1),'net-inspect-action');next.id='networkSourceNext';next.disabled=state.page>=count-1;
      pages.append(prev,next);detail.append(pages);
      detail.append(button('Return to review paths',returnToReview,'net-inspect-action'));
    }else if(n.source_refs.length){
      const b=button('Explore selected source facts',()=>sourceInspect(n.source_refs[0]),'net-inspect-action');b.id='networkSourceExplore';detail.append(b);
    }
    if(n.source_refs.length||n.queries.length||state.data.public_sources?.length){
      const source=el('details');source.append(el('summary','Evidence sources and attribution'));
      source.append(el('p','Snapshot '+state.data.snapshot),el('p',state.data.review_method));
      for(const s of (state.data.public_sources||[]).filter(s=>s.id===c.source_id)){
        const citation=el('p',s.creators+' ','net-source-citation'),link=el('a',s.title+' · version '+s.version);
        link.href=s.url;link.target='_blank';link.rel='noopener';citation.append(link);
        const license=el('a',s.license);license.href=s.license_url;license.target='_blank';license.rel='noopener';
        citation.append(document.createTextNode(' · '),license);source.append(citation,el('p',s.modifications));
      }
      source.append(el('p','Evidence identifiers preserve source traceability. This public selection excludes raw annotation-database records and internal policy objects.'));
      const records=el('div');
      function appendRefs(refs){for(const ref of refs){
        const b=button(ref.split('/').pop().slice(0,38)+'…',()=>sourceInspect(ref),'net-inspect-action');
        b.setAttribute('aria-label','Inspect source '+ref);records.append(b,el('p',ref,'net-source-id'));
      }}
      appendRefs(n.source_refs.slice(0,8));source.append(records);
      if(n.source_refs.length>8){const more=button('Show all '+n.source_refs.length+' source identifiers',()=>{appendRefs(n.source_refs.slice(8));more.remove();},'net-inspect-action');source.append(more);}
      detail.append(source);
    }
    detail.append(el('p',n.kind==='profile'?c.boundary:state.data.boundary,'net-boundary'));
    $('networkInspector').scrollTop=0;
  }
  function renderList() {
    const list=$('networkRecordList');list.replaceChildren();
    const c=current(),focus=getNode(state.focus)||getNode(c.root);
    const model=state.source?sourceModel():reviewModel();
    for(const n of model.nodes){
      const b=button('',()=>state.source?selectSourceNode(n):select(n.id),'net-record-row');b.dataset.state=n.state;
      b.dataset.recordId=n.id;
      b.setAttribute('aria-pressed',String(n.id===(state.source?state.sourceSelection:state.selected)));
      b.append(el('strong',n.label),el('span',labels[n.state]+' · '+(n.subtitle||n.kind)));list.append(b);
    }
    list.setAttribute('aria-label','Evidence records for '+focus.label);
  }
  function setView(view) {
    state.view=view;canvas.hidden=view!=='graph';$('networkRecordList').hidden=view!=='list';
    document.querySelectorAll('[data-network-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.networkView===view)));
    if(state.data){renderGraph();renderList();}
  }
  function search() {
    const term=$('networkSearch').value.trim().toLowerCase(),host=$('networkSearchResults');
    host.replaceChildren();host.hidden=!term;if(!term||!state.data)return;
    const matches=current().nodes.filter(n=>(n.label+' '+n.subtitle+' '+n.facts.flat().join(' ')).toLowerCase().includes(term));
    host.append(el('p',matches.length?matches.length+' matching records'+(matches.length>8?' · first 8 shown':''):'This case has no matching records. Try a gene ID, component label or depth.'));
    matches.slice(0,8).forEach(n=>host.append(button(n.label+' · '+labels[n.state],()=>select(n.id))));
    announce(matches.length+' matching records.');
  }
  function about() {
    const detail=$('networkDetail');detail.replaceChildren(el('p','Reading the evidence','net-eyebrow'),el('h3','A profile of review status'),
      el('p','Each color describes the state of an evidence record. The profile is qualitative; percentages, methane balances and risk scores require additional measurements.'),
      el('h4','Review provenance'),el('p',state.data.review_method),el('h4','Two linked views'),el('p',state.data.graph_contract),el('h4','What the geometry means'),el('p',state.data.layout_contract),
      el('h4','Scope'),el('p',state.data.scope),el('p','Snapshot '+state.data.snapshot),
      el('p','The earlier atlas scenes retain their August release. This explorer uses selected September source-linked cases, with all historical review restrictions preserved.'),
      el('p',state.data.boundary,'net-boundary'),button('Return to this case',()=>{render();$('networkAbout').focus({preventScroll:true});},'net-inspect-action'));
    const heading=detail.querySelector('h3');heading.tabIndex=-1;heading.focus();
    $('networkInspector').scrollTop=0;announce('How to read the evidence view.');
  }
  function immersive() {
    if(dialog.open){dialog.close();return;}
    priorFocus=document.activeElement;dialog.append(shell);dialog.showModal();
    $('networkImmersiveOpen').textContent='Return to page';document.body.style.overflow='hidden';
    requestAnimationFrame(()=>{fit();renderGraph();});
  }
  dialog.addEventListener('close',()=>{
    $('networkMount').append(shell);$('networkImmersiveOpen').textContent='Immersive view';document.body.style.overflow='';
    requestAnimationFrame(()=>{fit();renderGraph();priorFocus?.focus({preventScroll:true});});
  });
  async function load() {
    if(state.data||state.loading)return;
    state.loading=true;$('networkLoading').replaceChildren(el('p','Loading the source-linked evidence…'));
    try{
      const response=await fetch('data/molecular-evidence-network-public-v1.json');if(!response.ok)throw new Error('The evidence file is temporarily unavailable.');
      const data=await response.json();
      if(data.schema!=='mvo-case-explorer-public-1'||data.cases?.length!==3||!data.canonical_assertions)throw new Error('The evidence export needs review.');
      state.data=data;$('networkLoading').hidden=true;$('networkInterface').hidden=false;
      const host=$('networkCases');host.replaceChildren();
      data.cases.forEach((c,index)=>{const b=button('',()=>chooseCase(c.id),'net-case');b.dataset.networkCase=c.id;
        const text=el('span');text.append(el('strong',caseTitles[c.id]),el('small',c.record));
        b.append(el('span','0'+(index+1),'net-case-number'),text);host.append(b);});
      chooseCase(state.caseId);
    }catch(error){
      const host=$('networkLoading');host.hidden=false;host.replaceChildren(el('p',error.message),button('Retry evidence load',()=>load()));
      host.className='net-error';announce('Evidence could not be loaded. Retry is available.');
    }finally{state.loading=false;}
  }
  $('networkReset').addEventListener('click',()=>chooseCase(state.caseId));
  $('networkSearch').addEventListener('input',search);
  $('networkSearch').addEventListener('keydown',e=>{if(e.key==='Escape'){e.stopPropagation();$('networkSearch').value='';search();}});
  $('networkAbout').addEventListener('click',about);
  $('networkImmersiveOpen').addEventListener('click',immersive);
  $('networkZoomIn').addEventListener('click',()=>{state.zoom=Math.min(1.8,state.zoom+.15);transform();});
  $('networkZoomOut').addEventListener('click',()=>{state.zoom=Math.max(.65,state.zoom-.15);transform();});
  $('networkFit').addEventListener('click',fit);
  $('networkPagePrevious').addEventListener('click',()=>changePage(-1));
  $('networkPageNext').addEventListener('click',()=>changePage(1));
  document.querySelectorAll('[data-network-view]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.networkView)));
  let drag=null;
  canvas.addEventListener('pointerdown',e=>{if(e.pointerType!=='mouse'||e.target.closest('button,.net-edge-hit'))return;drag={x:e.clientX,y:e.clientY,pan:{...state.pan}};canvas.setPointerCapture(e.pointerId);canvas.style.cursor='grabbing';});
  canvas.addEventListener('pointermove',e=>{if(!drag)return;state.pan={x:drag.pan.x+e.clientX-drag.x,y:drag.pan.y+e.clientY-drag.y};transform();});
  function release(){drag=null;canvas.style.cursor='';}
  canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);
  new ResizeObserver(()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>{if(state.data)renderGraph();});}).observe(canvas);
  // Text-only zoom does not resize the viewport. Observe a text-sized probe so
  // native browser enlargement can reflow fixed graph and page-chrome geometry.
  const textProbe=el('span','M','eb-text-probe');textProbe.setAttribute('aria-hidden','true');document.body.append(textProbe);
  new ResizeObserver(()=>{
    const enlarged=parseFloat(getComputedStyle(textProbe).fontSize)>20;
    if(document.documentElement.classList.contains('eb-text-enlarged')!==enlarged){
      document.documentElement.classList.toggle('eb-text-enlarged',enlarged);
      if(state.data){fit();renderGraph();}
    }
  }).observe(textProbe);
  document.addEventListener('emergentbiome:network-case',e=>{
    if(['interpret','locate','design'].includes(e.detail?.caseId)){chooseCase(e.detail.caseId);section.scrollIntoView({behavior:reduced.matches?'auto':'smooth'});}
  });
  const observer=new IntersectionObserver(entries=>{if(entries.some(e=>e.isIntersecting)){load();observer.disconnect();}},{rootMargin:'400px'});observer.observe(section);
  if(location.hash==='#scene-network')load();
  window.EBNetwork={getState:()=>({caseId:state.caseId,focus:state.focus,selected:state.selected,source:state.source,view:state.view,loaded:!!state.data,visibleNodes:activeNodes.length}),selectCase:chooseCase};
})();
