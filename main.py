<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;500;600;700;800&family=Barlow+Condensed:wght@400;600;700;800&display=swap');

  :root {
    --black:      #080808;
    --surface:    #111111;
    --surface2:   #181818;
    --border:     rgba(255,255,255,0.07);
    --border-hi:  rgba(255,255,255,0.13);
    --text:       #f2f2f2;
    --muted:      rgba(242,242,242,0.45);
    --dim:        rgba(242,242,242,0.22);
    --gold:       #d4b483;
    --gold-glow:  rgba(212,180,131,0.10);
    --red:        #ff4d4d;
    --orange:     #ff9940;
    --blue:       #4fb3ff;
    --green:      #3ddc84;
    --red-bg:     rgba(255,77,77,0.07);
    --orange-bg:  rgba(255,153,64,0.07);
    --blue-bg:    rgba(79,179,255,0.07);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  .cm {
    font-family: 'Barlow', sans-serif;
    background: var(--black);
    color: var(--text);
    -webkit-font-smoothing: antialiased;
  }

  /* ── HEADER ── */
  .cm-header {
    padding: 64px 24px 48px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .cm-header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 70% 55% at 50% 0%, rgba(212,180,131,0.08) 0%, transparent 65%);
    pointer-events: none;
  }
  .cm-logo {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 22px;
    opacity: 0;
    animation: fadeUp .5s .05s ease forwards;
  }
  .cm-logo span {
    display: inline-block;
    width: 5px; height: 5px;
    background: var(--gold);
    border-radius: 50%;
    vertical-align: middle;
    margin: 0 10px 2px;
    box-shadow: 0 0 8px var(--gold);
  }
  .cm-headline {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: clamp(36px, 6.5vw, 68px);
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -0.01em;
    text-transform: uppercase;
    margin-bottom: 16px;
    opacity: 0;
    animation: fadeUp .55s .12s ease forwards;
  }
  .cm-headline .line2 {
    display: block;
    color: var(--gold);
  }
  .cm-desc {
    max-width: 520px;
    margin: 0 auto 40px;
    font-size: 15px;
    font-weight: 300;
    line-height: 1.65;
    color: var(--muted);
    opacity: 0;
    animation: fadeUp .55s .2s ease forwards;
  }
  .cm-desc strong { color: var(--text); font-weight: 500; }

  /* ── UPLOAD ZONE ── */
  .cm-upload-wrap {
    max-width: 680px;
    margin: 0 auto;
    opacity: 0;
    animation: fadeUp .55s .28s ease forwards;
  }
  .cm-drop-zone {
    position: relative;
    border: 1.5px dashed var(--border-hi);
    border-radius: 14px;
    padding: 44px 32px;
    cursor: pointer;
    background: rgba(255,255,255,0.012);
    transition: border-color .2s, background .2s, transform .2s;
    text-align: center;
    margin-bottom: 20px;
  }
  .cm-drop-zone input[type=file] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
  }
  .cm-drop-zone:hover, .cm-drop-zone.drag-on {
    border-color: var(--gold);
    background: var(--gold-glow);
    transform: translateY(-2px);
  }
  .cm-drop-icon {
    width: 40px; height: 40px; margin: 0 auto 14px; color: var(--gold); opacity: .65;
  }
  .cm-drop-title { font-size: 15px; font-weight: 600; margin-bottom: 5px; }
  .cm-drop-sub   { font-size: 13px; color: var(--muted); }

  /* ── ANALYZE BUTTON ── */
  .cm-btn {
    width: 100%;
    display: flex; align-items: center; justify-content: center; gap: 9px;
    background: var(--gold);
    color: #1a1208;
    border: none;
    padding: 16px 28px;
    font-family: 'Barlow', sans-serif;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border-radius: 10px;
    cursor: pointer;
    transition: transform .18s, box-shadow .18s, opacity .18s;
    box-shadow: 0 0 28px rgba(212,180,131,.16), 0 4px 14px rgba(0,0,0,.4);
    position: relative; overflow: hidden;
  }
  .cm-btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 0 42px rgba(212,180,131,.26), 0 8px 20px rgba(0,0,0,.5); }
  .cm-btn:active:not(:disabled) { transform: scale(.98); }
  .cm-btn:disabled { opacity: .45; cursor: not-allowed; }
  .cm-btn-shine {
    position: absolute; top: 0; left: -100%; width: 55%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.22), transparent);
    transform: skewX(-18deg);
  }
  .cm-btn:hover:not(:disabled) .cm-btn-shine { animation: shine .55s ease forwards; }
  @keyframes shine { to { left: 150%; } }

  /* ── STATUS ── */
  .cm-status {
    margin-top: 18px; min-height: 20px;
    display: flex; align-items: center; justify-content: center; gap: 8px;
    font-size: 13px; color: var(--muted);
  }
  .cm-spin {
    width: 15px; height: 15px; flex-shrink: 0;
    border: 2px solid rgba(212,180,131,.18); border-top-color: var(--gold);
    border-radius: 50%; animation: spin .65s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── RESULTS ── */
  .cm-results { max-width: 720px; margin: 0 auto; padding: 0 20px 80px; display: none; }
  .cm-results.on { display: block; }

  /* ── UPLOADED IMAGE ── */
  .cm-image-block {
    margin-bottom: 24px;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--border-hi);
    background: var(--surface);
    position: relative;
  }
  .cm-image-block img {
    width: 100%; display: block;
    max-height: 480px; object-fit: contain;
    background: #0d0d0d;
  }
  .cm-image-label {
    position: absolute; bottom: 12px; left: 12px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 11px; font-weight: 700; letter-spacing: .15em; text-transform: uppercase;
    color: var(--muted);
    background: rgba(0,0,0,.55); backdrop-filter: blur(6px);
    padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border);
  }

  /* ── SCORE ── */
  .cm-score-card {
    background: var(--surface); border: 1px solid var(--border-hi);
    border-radius: 16px; padding: 36px 28px; text-align: center;
    margin-bottom: 16px; position: relative; overflow: hidden;
  }
  .cm-score-card::before {
    content: ''; position: absolute; top: -50px; left: 50%; transform: translateX(-50%);
    width: 260px; height: 160px;
    background: radial-gradient(ellipse, rgba(212,180,131,.09) 0%, transparent 70%);
    pointer-events: none;
  }
  .cm-score-badge {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 11px; font-weight: 700; letter-spacing: .2em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 10px;
  }
  .cm-score-scene {
    font-size: 11px; font-weight: 500; letter-spacing: .12em; text-transform: uppercase;
    color: var(--gold); margin-bottom: 12px; opacity: .75;
  }
  .cm-score-num {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: clamp(72px,14vw,100px); font-weight: 800;
    line-height: 1; letter-spacing: -.02em; color: var(--gold);
  }
  .cm-score-num sup { font-size: .32em; color: var(--muted); vertical-align: top; margin-top: .55em; font-weight: 600; }
  .cm-score-bar-wrap { max-width: 300px; margin: 20px auto 0; height: 3px; background: rgba(255,255,255,.05); border-radius: 2px; overflow: hidden; }
  .cm-score-bar { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--gold), #fff0d0); width: 0%; transition: width 1s cubic-bezier(.16,1,.3,1); box-shadow: 0 0 10px rgba(212,180,131,.5); }

  /* ── BREAKDOWN ── */
  .cm-section-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 11px; font-weight: 700; letter-spacing: .2em; text-transform: uppercase;
    color: var(--dim); margin-bottom: 12px; padding-left: 2px;
  }
  .cm-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(150px,1fr));
    gap: 10px; margin-bottom: 16px;
  }
  .cm-metric {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 14px;
    opacity: 0; transform: translateY(10px);
    transition: opacity .4s, transform .4s, border-color .2s;
  }
  .cm-metric.on { opacity: 1; transform: none; }
  .cm-metric:hover { border-color: var(--border-hi); }
  .cm-metric-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 10px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 8px;
  }
  .cm-metric-val {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 34px; font-weight: 800; line-height: 1; letter-spacing: -.01em;
    margin-bottom: 9px;
  }
  .cm-bar-wrap { height: 3px; background: rgba(255,255,255,.06); border-radius: 2px; overflow: hidden; }
  .cm-bar { height: 100%; border-radius: 2px; width: 0%; transition: width .85s cubic-bezier(.16,1,.3,1); }

  .hi .cm-metric-val { color: var(--green); }
  .hi .cm-bar        { background: var(--green); box-shadow: 0 0 7px rgba(61,220,132,.4); }
  .md .cm-metric-val { color: var(--orange); }
  .md .cm-bar        { background: var(--orange); box-shadow: 0 0 7px rgba(255,153,64,.4); }
  .lo .cm-metric-val { color: var(--red); }
  .lo .cm-bar        { background: var(--red); box-shadow: 0 0 7px rgba(255,77,77,.4); }

  /* ── DIVIDER ── */
  .cm-div { height: 1px; background: var(--border); margin: 24px 0; }

  /* ── SUGGESTIONS ── */
  .cm-sug-card {
    display: flex; gap: 14px; align-items: flex-start;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px; margin-bottom: 10px;
    opacity: 0; transform: translateY(8px);
    transition: opacity .4s, transform .4s, border-color .2s;
  }
  .cm-sug-card.on { opacity: 1; transform: none; }
  .cm-sug-card:hover { border-color: var(--border-hi); }
  .cm-sug-card.p-high   { border-left: 2px solid var(--red);    background: linear-gradient(135deg, var(--red-bg) 0%, var(--surface) 45%); }
  .cm-sug-card.p-medium { border-left: 2px solid var(--orange); background: linear-gradient(135deg, var(--orange-bg) 0%, var(--surface) 45%); }
  .cm-sug-card.p-low    { border-left: 2px solid var(--blue);   background: linear-gradient(135deg, var(--blue-bg) 0%, var(--surface) 45%); }
  .cm-sug-icon {
    flex-shrink: 0; width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center; font-size: 15px;
  }
  .p-high   .cm-sug-icon { background: var(--red-bg);    color: var(--red); }
  .p-medium .cm-sug-icon { background: var(--orange-bg); color: var(--orange); }
  .p-low    .cm-sug-icon { background: var(--blue-bg);   color: var(--blue); }
  .cm-sug-body { flex: 1; }
  .cm-sug-top { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; flex-wrap: wrap; }
  .cm-sug-cat { font-size: 13px; font-weight: 700; color: var(--text); }
  .cm-sug-badge {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 10px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
    padding: 2px 8px; border-radius: 100px;
  }
  .p-high   .cm-sug-badge { background: var(--red-bg);    color: var(--red);    border: 1px solid rgba(255,77,77,.2); }
  .p-medium .cm-sug-badge { background: var(--orange-bg); color: var(--orange); border: 1px solid rgba(255,153,64,.2); }
  .p-low    .cm-sug-badge { background: var(--blue-bg);   color: var(--blue);   border: 1px solid rgba(79,179,255,.2); }
  .cm-sug-msg { font-size: 13px; font-weight: 300; line-height: 1.6; color: rgba(242,242,242,.7); }

  /* ── RAW JSON ── */
  .cm-raw-btn {
    display: inline-flex; align-items: center; gap: 6px;
    background: none; border: 1px solid var(--border); color: var(--muted);
    font-family: 'Barlow', sans-serif; font-size: 12px; font-weight: 500;
    padding: 7px 14px; border-radius: 8px; cursor: pointer;
    transition: border-color .2s, color .2s; margin-bottom: 8px;
  }
  .cm-raw-btn:hover { border-color: var(--border-hi); color: var(--text); }
  .cm-raw-box {
    display: none; background: var(--surface2); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px; font-size: 11px;
    font-family: 'SF Mono','Fira Code',monospace; color: rgba(242,242,242,.45);
    white-space: pre-wrap; line-height: 1.6; max-height: 280px; overflow-y: auto;
  }
  .cm-raw-box.on { display: block; }

  /* ── ANIMATIONS ── */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: none; }
  }

  /* ── MOBILE ── */
  @media (max-width: 500px) {
    .cm-header { padding: 48px 16px 36px; }
    .cm-grid { grid-template-columns: repeat(2,1fr); gap: 8px; }
    .cm-metric { padding: 13px 11px; }
    .cm-sug-card { padding: 14px; }
    .cm-image-block img { max-height: 300px; }
  }
</style>

<div class="cm">

  <!-- HEADER -->
  <div class="cm-header">
    <div class="cm-logo">Cine<span></span>Mind</div>
    <h1 class="cm-headline">
      Analyze your shot.<br>
      <span class="line2">Grade with confidence.</span>
    </h1>
    <p class="cm-desc">
      Upload a <strong>Rec.709 preview screenshot</strong> from your Apple Log footage.
      CineMind breaks down exposure, color, sharpness, and cinematography —
      then tells you exactly what to fix before you grade.
    </p>

    <div class="cm-upload-wrap">
      <div class="cm-drop-zone" id="cmDrop">
        <input type="file" id="cmFile" accept="image/*">
        <svg class="cm-drop-icon" viewBox="0 0 40 40" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="3" width="34" height="34" rx="7"/>
          <path d="M20 26V14M14 20l6-6 6 6"/>
          <path d="M12 32h16" opacity=".4"/>
        </svg>
        <div class="cm-drop-title">Drop your screenshot here</div>
        <div class="cm-drop-sub">or click to browse &nbsp;·&nbsp; JPG, PNG, HEIC</div>
      </div>

      <button class="cm-btn" id="cmBtn" onclick="cmRun()">
        <span class="cm-btn-shine"></span>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="8" cy="8" r="6.5"/>
          <path d="M8 5.5v2.8l1.8 1.1"/>
        </svg>
        Analyze Shot
      </button>

      <div class="cm-status" id="cmStatus"></div>
    </div>
  </div>

  <!-- RESULTS -->
  <div class="cm-results" id="cmResults">

    <!-- Uploaded image -->
    <div class="cm-image-block">
      <img id="cmPreviewImg" src="" alt="Uploaded shot">
      <div class="cm-image-label" id="cmImgLabel">—</div>
    </div>

    <!-- Score -->
    <div class="cm-score-card">
      <div class="cm-score-badge">Cinematic Score</div>
      <div class="cm-score-scene" id="cmSceneTag"></div>
      <div class="cm-score-num" id="cmScoreNum">—<sup>/100</sup></div>
      <div class="cm-score-bar-wrap"><div class="cm-score-bar" id="cmScoreBar"></div></div>
    </div>

    <!-- Breakdown -->
    <div class="cm-section-label">Breakdown</div>
    <div class="cm-grid" id="cmGrid"></div>

    <div class="cm-div"></div>

    <!-- Suggestions -->
    <div id="cmSugSection" style="display:none">
      <div class="cm-section-label">How to improve your shot</div>
      <div id="cmSugList"></div>
      <div class="cm-div"></div>
    </div>

    <!-- Raw JSON -->
    <button class="cm-raw-btn" onclick="cmToggleRaw()">
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
        <path d="M4 2L1 6l3 4M8 2l3 4-3 4"/>
      </svg>
      Raw JSON
    </button>
    <div class="cm-raw-box" id="cmRaw"></div>

  </div>

</div>

<script>
const API = "https://cinemind-api-4052.onrender.com/analyze";

const ICONS = {
  exposure:"☀", contrast:"◑", color:"◈", skin:"◉",
  noise:"∿", sharpness:"◎", cinematography:"▣", composition:"⬡"
};
const SUG_ICONS = { Exposure:"☀", Color:"◈", Sharpness:"◎", Cinematography:"▣", Composition:"⬡", Noise:"∿" };

function scoreClass(v) { return v>=70?"hi":v>=45?"md":"lo"; }

// ── Drag & drop ──
const drop = document.getElementById("cmDrop");
const fileInput = document.getElementById("cmFile");
drop.addEventListener("dragover",  e=>{e.preventDefault();drop.classList.add("drag-on");});
drop.addEventListener("dragleave", ()=>drop.classList.remove("drag-on"));
drop.addEventListener("drop", e=>{
  e.preventDefault(); drop.classList.remove("drag-on");
  const f = e.dataTransfer.files[0];
  if(f&&f.type.startsWith("image/")){
    const dt=new DataTransfer(); dt.items.add(f); fileInput.files=dt.files;
  }
});

// ── Animated counter ──
function animateNum(target){
  const el=document.getElementById("cmScoreNum");
  const bar=document.getElementById("cmScoreBar");
  const dur=900; const t0=performance.now();
  function step(now){
    const t=Math.min((now-t0)/dur,1);
    const e=1-Math.pow(1-t,3);
    el.innerHTML=`${Math.round(e*target)}<sup>/100</sup>`;
    if(t<1)requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
  setTimeout(()=>{bar.style.width=target+"%";},60);
}

// ── Build breakdown grid ──
function buildGrid(bd){
  const grid=document.getElementById("cmGrid");
  grid.innerHTML="";
  Object.entries(bd).forEach(([key,val],i)=>{
    const cls=scoreClass(val);
    const card=document.createElement("div");
    card.className=`cm-metric ${cls}`;
    card.style.transitionDelay=`${i*55}ms`;
    card.innerHTML=`
      <div class="cm-metric-label">${ICONS[key]||"·"} ${key.charAt(0).toUpperCase()+key.slice(1)}</div>
      <div class="cm-metric-val">${Math.round(val)}</div>
      <div class="cm-bar-wrap"><div class="cm-bar" data-v="${val}"></div></div>
    `;
    grid.appendChild(card);
    requestAnimationFrame(()=>setTimeout(()=>card.classList.add("on"),i*55+40));
  });
  setTimeout(()=>{
    grid.querySelectorAll(".cm-bar").forEach(b=>b.style.width=b.dataset.v+"%");
  },180);
}

// ── Build suggestions ──
function buildSuggestions(sugs){
  const sec=document.getElementById("cmSugSection");
  const list=document.getElementById("cmSugList");
  if(!sugs||!sugs.length){sec.style.display="none";return;}
  list.innerHTML="";
  sugs.forEach((s,i)=>{
    const card=document.createElement("div");
    card.className=`cm-sug-card p-${s.priority}`;
    card.style.transitionDelay=`${i*70}ms`;
    card.innerHTML=`
      <div class="cm-sug-icon">${SUG_ICONS[s.category]||"·"}</div>
      <div class="cm-sug-body">
        <div class="cm-sug-top">
          <span class="cm-sug-cat">${s.category}</span>
          <span class="cm-sug-badge">${s.priority}</span>
        </div>
        <div class="cm-sug-msg">${s.message}</div>
      </div>
    `;
    list.appendChild(card);
    requestAnimationFrame(()=>setTimeout(()=>card.classList.add("on"),i*70+100));
  });
  sec.style.display="block";
}

// ── Main ──
async function cmRun(){
  const file=fileInput.files[0];
  if(!file){alert("Please upload a screenshot first.");return;}

  const btn=document.getElementById("cmBtn");
  const status=document.getElementById("cmStatus");
  const results=document.getElementById("cmResults");

  btn.disabled=true;
  results.classList.remove("on");
  status.innerHTML=`<div class="cm-spin"></div> Waking up API…`;

  // Show the preview image immediately
  const reader=new FileReader();
  reader.onload=e=>{
    document.getElementById("cmPreviewImg").src=e.target.result;
    document.getElementById("cmImgLabel").textContent=file.name;
  };
  reader.readAsDataURL(file);

  const fd=new FormData(); fd.append("file",file);

  try {
    status.innerHTML=`<div class="cm-spin"></div> Analyzing…`;
    const res=await fetch(API,{method:"POST",body:fd});
    const data=await res.json();
    if(!data.ok)throw new Error(data.error||"API error");

    status.innerHTML="";
    results.classList.add("on");

    // Scene tag
    const sceneTag=document.getElementById("cmSceneTag");
    sceneTag.textContent=data.scene_type==="subject"?"— Subject scene —":"— Ambient scene —";

    animateNum(data.score);
    if(data.breakdown) buildGrid(data.breakdown);
    if(data.suggestions) buildSuggestions(data.suggestions);
    document.getElementById("cmRaw").textContent=JSON.stringify(data,null,2);

    setTimeout(()=>results.scrollIntoView({behavior:"smooth",block:"start"}),120);
  }catch(err){
    console.error(err);
    status.innerHTML=`<span style="color:var(--red)">Analysis failed — ${err.message}</span>`;
  }finally{
    btn.disabled=false;
  }
}

function cmToggleRaw(){ document.getElementById("cmRaw").classList.toggle("on"); }
</script>
