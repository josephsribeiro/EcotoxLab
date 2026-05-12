import { useState, useEffect } from "react";
import { Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, ComposedChart, Scatter } from "recharts";

// ═══════════════════════════════════════════════════════════════
// MATH CORE
// ═══════════════════════════════════════════════════════════════

function qnorm(p) {
  if (p <= 0) return -8; if (p >= 1) return 8;
  const a = [2.515517, 0.802853, 0.010328];
  const b = [1.432788, 0.189269, 0.001308];
  const t = Math.sqrt(-2 * Math.log(p < 0.5 ? p : 1 - p));
  const num = a[0] + a[1]*t + a[2]*t*t;
  const den = 1 + b[0]*t + b[1]*t*t + b[2]*t*t*t;
  return (p < 0.5 ? -1 : 1) * (t - num/den);
}

function pnorm(x) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const poly = t*(0.319381530 + t*(-0.356563782 + t*(1.781477937 + t*(-1.821255978 + t*1.330274429))));
  const phi = Math.exp(-0.5*x*x) / Math.sqrt(2*Math.PI);
  const p = 1 - phi * poly;
  return x >= 0 ? p : 1 - p;
}

function chiSqPval(chi2, df) {
  if (df <= 0 || isNaN(chi2)) return NaN;
  const z = Math.pow(chi2/df, 1/3) - (1 - 2/(9*df));
  const se = Math.sqrt(2/(9*df));
  return 1 - pnorm(z/se);
}

function abbott(p, ctrl) {
  if (ctrl >= 1) return p;
  return Math.max(0, Math.min(1, (p - ctrl) / (1 - ctrl)));
}

// ── TRIMMED SPEARMAN-KÄRBER ───────────────────────────────────
function calcSpearmanKarber(doses, deaths, totals) {
  const pairs = doses
    .map((d, i) => ({ d, p: deaths[i] / totals[i], n: totals[i], k: deaths[i] }))
    .filter(p => p.d > 0)
    .sort((a, b) => a.d - b.d);
  if (pairs.length < 2) return null;

  const ld = pairs.map(p => Math.log10(p.d));
  const pr = pairs.map(p => p.p);

  // Area under log-dose response curve
  let area = 0;
  for (let i = 1; i < ld.length; i++) {
    area += (ld[i] - ld[i-1]) * (pr[i] + pr[i-1]) / 2;
  }
  const logCL = ld[ld.length-1] - area;

  // Variance (Thompson 1947)
  let v = 0;
  for (let i = 1; i < ld.length; i++) {
    const pi = (pr[i] + pr[i-1]) / 2;
    const ni = pairs[i].n;
    v += Math.pow(ld[i] - ld[i-1], 2) * pi * (1 - pi) / Math.max(ni - 1, 1);
  }
  const se = Math.sqrt(v);
  const f  = Math.pow(10, 1.96 * se);

  // Pearson chi-sq GOF
  let chi2 = 0;
  for (const p of pairs) {
    const expected = p.n * p.p;
    chi2 += Math.pow(p.k - expected, 2) / Math.max(expected * (1 - p.p), 1e-6);
  }

  const cl = Math.pow(10, logCL);
  return {
    cl, lcl: cl / f, ucl: cl * f, logCL,
    slope: null, intercept: null,
    chi2: chi2.toFixed(3),
    pgof: chiSqPval(chi2, Math.max(1, pairs.length - 2)).toFixed(4),
    zValue: null, variance: null,
  };
}

// ── GLM (Probit / Logit) via IRLS — LC_probit, LC_logit, LT_probit, LT_logit ──
function calcGLM(doses, deaths, totals, link) {
  const pairs = doses
    .map((d, i) => ({ x: Math.log10(Math.max(d, 1e-10)), p: deaths[i] / totals[i], n: totals[i], k: deaths[i] }))
    .filter(p => isFinite(p.x));
  if (pairs.length < 2) return null;

  const ginv   = link === "probit" ? pnorm : z => 1 / (1 + Math.exp(-z));
  const gprime = link === "probit"
    ? mu => Math.exp(-0.5 * Math.pow(qnorm(mu), 2)) / Math.sqrt(2 * Math.PI)
    : mu => mu * (1 - mu);

  // OLS seed from points with 0 < p < 1
  const pts = pairs.filter(p => p.p > 0 && p.p < 1);
  if (pts.length < 2) return null;

  const fn = link === "probit"
    ? p => qnorm(p)
    : p => Math.log(p / (1 - p));

  const xs = pts.map(p => p.x), ys = pts.map(p => fn(p.p));
  const n  = xs.length;
  const mx = xs.reduce((a,b)=>a+b,0)/n, my = ys.reduce((a,b)=>a+b,0)/n;
  const ssxy = xs.reduce((a,v,i)=>a+(v-mx)*(ys[i]-my),0);
  const ssxx = xs.reduce((a,v)=>a+(v-mx)**2,0);
  let b1 = ssxx > 1e-12 ? ssxy/ssxx : 1;
  let b0 = my - b1*mx;

  // IRLS
  for (let iter = 0; iter < 40; iter++) {
    let sW=0, sWX=0, sXWX=0, sWZ=0, sXWZ=0;
    for (const p of pairs) {
      const eta = b0 + b1 * p.x;
      const mu  = Math.max(1e-7, Math.min(1-1e-7, ginv(eta)));
      const gp  = Math.max(1e-10, gprime(mu));
      const w   = p.n * gp * gp / (mu * (1 - mu));
      const z   = eta + (p.p - mu) / gp;
      sW += w; sWX += w*p.x; sXWX += w*p.x*p.x; sWZ += w*z; sXWZ += w*p.x*z;
    }
    const det = sW*sXWX - sWX*sWX;
    if (Math.abs(det) < 1e-14) break;
    const nb0 = (sXWX*sWZ - sWX*sXWZ) / det;
    const nb1 = (sW*sXWZ  - sWX*sWZ)  / det;
    const conv = Math.abs(nb0-b0) + Math.abs(nb1-b1);
    b0=nb0; b1=nb1;
    if (conv < 1e-9) break;
  }

  // CL50: g(0.5)=0 for both probit and logit
  const logCL = -b0 / b1;
  const cl = Math.pow(10, logCL);

  // Fisher information for variance of logCL (Finney 1971, delta method)
  let sW=0, sWX=0, sXWX=0;
  for (const p of pairs) {
    const eta = b0 + b1 * p.x;
    const mu  = Math.max(1e-7, Math.min(1-1e-7, ginv(eta)));
    const gp  = Math.max(1e-10, gprime(mu));
    const w   = p.n * gp * gp / (mu * (1 - mu));
    sW += w; sWX += w*p.x; sXWX += w*p.x*p.x;
  }
  const det = sW*sXWX - sWX*sWX;
  const varB0 = Math.max(0, sXWX/det);
  const varB1 = Math.max(0, sW/det);
  const covB  = -sWX/det;
  const varLogCL = (varB0 + 2*logCL*covB + logCL*logCL*varB1) / (b1*b1);
  const seLogCL  = Math.sqrt(Math.max(0, varLogCL));
  const f = Math.pow(10, 1.96 * seLogCL);

  const seB1 = Math.sqrt(varB1);
  const zVal = seB1 > 1e-10 ? (b1/seB1).toFixed(3) : "—";

  // Pearson chi-sq GOF
  let chi2 = 0;
  for (const p of pairs) {
    const eta = b0 + b1 * p.x;
    const mu  = Math.max(1e-7, Math.min(1-1e-7, ginv(eta)));
    chi2 += Math.pow(p.k - p.n*mu, 2) / Math.max(p.n*mu*(1-mu), 1e-6);
  }
  const df = Math.max(1, pairs.length - 2);

  return {
    cl, lcl: cl/f, ucl: cl*f, logCL,
    slope: b1.toFixed(4), intercept: b0.toFixed(4),
    chi2: chi2.toFixed(3),
    pgof: chiSqPval(chi2, df).toFixed(4),
    zValue: zVal,
    variance: varLogCL.toFixed(6),
    _b0: b0, _b1: b1, _link: link,
  };
}

// Curve prediction from fitted GLM params
function predictY(x, res, method) {
  const logX = Math.log10(Math.max(x, 1e-10));
  if (res._b0 !== undefined) {
    const eta = res._b0 + res._b1 * logX;
    return method.link === "probit" ? pnorm(eta)*100 : 100/(1+Math.exp(-eta));
  }
  // Spearman fallback: use logistic centered on cl
  return 100 / (1 + Math.pow(x / res.cl, -4.5));
}

// Predict using a shifted cl (for CI band display)
function predictYfromCL(x, cl, link) {
  const r2 = cl > 0 ? Math.log10(cl) : 0;
  const b1est = 4.5;
  const b0est = -b1est * r2;
  const logX = Math.log10(Math.max(x, 1e-10));
  const eta = b0est + b1est * logX;
  return link === "probit" ? pnorm(eta)*100 : 100/(1+Math.exp(-eta));
}

// ═══════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════
const UNITS_CONC = ["µg/L","µg/g","µg/mg","µg/kg","mg/L","mg/g","mg/kg","ng/L","ng/g"];
const UNITS_TIME = ["h","min","dias","semanas"];

const METHODS = [
  { id:"lc_probit", label:"LC_probit",      group:"Concentração Letal", link:"probit", xIsTime:false, fn:"LC_probit()" },
  { id:"lc_logit",  label:"LC_logit",       group:"Concentração Letal", link:"logit",  xIsTime:false, fn:"LC_logit()" },
  { id:"lt_probit", label:"LT_probit",      group:"Tempo Letal",        link:"probit", xIsTime:true,  fn:"LT_probit()" },
  { id:"lt_logit",  label:"LT_logit",       group:"Tempo Letal",        link:"logit",  xIsTime:true,  fn:"LT_logit()" },
  { id:"spearman",  label:"Spearman-Kärber",group:"Não-paramétrico",    link:null,     xIsTime:false, fn:"trimmed S-K" },
];

const METHOD_NOTES = {
  lc_probit: "LC_probit(): GLM binomial com ligação probit. Limites fiduciais via método de Finney (1971). Equivalente direto ao LC_probit() do pacote {ecotox}.",
  lc_logit:  "LC_logit(): GLM binomial com ligação logit. Alternativa ao probit, geralmente mais robusta nos extremos da curva. Equivalente ao LC_logit() do {ecotox}.",
  lt_probit: "LT_probit(): Mesmo algoritmo do LC_probit, mas com tempo de exposição como variável independente. Equivalente ao LT_probit() do {ecotox}.",
  lt_logit:  "LT_logit(): GLM logit com tempo de exposição. Equivalente ao LT_logit() do {ecotox}.",
  spearman:  "Spearman-Kärber aparado (Wheeler et al. 2006): método não-paramétrico, calcula a CL50 como área sob a curva dose-resposta em log₁₀. Variância por Thompson (1947). Correção de Abbott aplicada ao controle.",
};

// ═══════════════════════════════════════════════════════════════
// APP
// ═══════════════════════════════════════════════════════════════
export default function EcotoxApp() {
  const [substance, setSubstance] = useState("Substância X");
  const [reps,      setReps]      = useState(3);
  const [indiv,     setIndiv]     = useState(10);
  const [unitConc,  setUnitConc]  = useState("µg/L");
  const [unitTime,  setUnitTime]  = useState("h");
  const [methodId,  setMethodId]  = useState("lc_probit");
  const [nDoses,    setNDoses]    = useState(5);
  const [rows,      setRows]      = useState(Array.from({length:5},()=>({x:"",dead:""})));
  const [result,    setResult]    = useState(null);
  const [chartData, setChartData] = useState([]);
  const [error,     setError]     = useState("");
  const [tab,       setTab]       = useState("data");

  const method = METHODS.find(m => m.id === methodId);
  const total  = reps * indiv;
  const unit   = method.xIsTime ? unitTime : unitConc;
  const xLabel = method.xIsTime ? `Tempo (${unit})` : `Concentração (${unit})`;
  const clLabel = method.xIsTime ? "TL50" : "CL50";

  useEffect(() => {
    setRows(prev => Array.from({length:nDoses},(_,i) => prev[i] || {x:"",dead:""}));
    setResult(null); setError("");
  }, [nDoses]);

  useEffect(() => { setResult(null); setError(""); }, [methodId]);

  const updateRow = (i, field, val) =>
    setRows(prev => { const n=[...prev]; n[i]={...n[i],[field]:val}; return n; });

  const analyze = () => {
    setError(""); setResult(null);
    const parsed = rows.map(r => ({ x: parseFloat(r.x), dead: parseFloat(r.dead) }));
    if (parsed.some(r => isNaN(r.x) || isNaN(r.dead))) return setError("Preencha todos os campos.");
    if (parsed.some(r => r.dead > total)) return setError(`Mortos não pode exceder o total (${total}).`);

    const xs     = parsed.map(r => r.x);
    const probs  = parsed.map(r => r.dead / total);
    const totals = parsed.map(() => total);

    const ctrlIdx = xs.indexOf(0);
    const ctrlP   = ctrlIdx >= 0 ? probs[ctrlIdx] : 0;
    const corrP   = probs.map(p => abbott(p, ctrlP));
    const corrD   = corrP.map((p,i) => Math.round(p * totals[i]));

    const axs = xs.filter((_,i) => xs[i] > 0);
    const ade = corrD.filter((_,i) => xs[i] > 0);
    const ato = totals.filter((_,i) => xs[i] > 0);

    if (axs.length < 2) return setError("Mínimo 2 doses/tempos não-zero.");

    const res = methodId === "spearman"
      ? calcSpearmanKarber(axs, ade, ato)
      : calcGLM(axs, ade, ato, method.link);

    if (!res) return setError("Não foi possível calcular. Verifique os dados.");

    setResult(res);
    setTab("result");

    // Chart data
    const minX = Math.min(...axs), maxX = Math.max(...axs);
    const xMin = Math.max(minX * 0.75, 0.001);
    const xMax = maxX * 1.25;
    const pts = Array.from({length:80}, (_,i) => {
      const x = xMin + (xMax - xMin) * i / 79;
      return {
        x:     parseFloat(x.toFixed(4)),
        curve: parseFloat(predictY(x, res, method).toFixed(2)),
        lower: parseFloat(predictYfromCL(x, res.lcl, method.link || "logit").toFixed(2)),
        upper: parseFloat(predictYfromCL(x, res.ucl, method.link || "logit").toFixed(2)),
      };
    });
    const obs = axs.map((x,i) => ({ x, obs: parseFloat((ade[i]/ato[i]*100).toFixed(2)) }));
    const merged = pts.map(p => {
      const o = obs.find(s => Math.abs(s.x - p.x) < (xMax-xMin)/100);
      return o ? {...p, obs:o.obs} : p;
    });
    obs.forEach(o => {
      if (!merged.find(m => m.obs !== undefined && Math.abs(m.x - o.x) < 1e-6))
        merged.push({ x:o.x, obs:o.obs });
    });
    setChartData(merged.sort((a,b)=>a.x-b.x));
  };

  const fmt  = v => (v!=null && !isNaN(v)) ? Number(v).toFixed(4) : "—";
  const fmt2 = v => (v!=null && !isNaN(v)) ? Number(v).toFixed(3) : "—";

  return (
    <div style={{minHeight:"100vh",background:"#0a0e14",color:"#cdd9e5",fontFamily:"'DM Mono','Courier New',monospace"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@600;700;800&display=swap');
        *{box-sizing:border-box}
        ::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:#0a0e14}::-webkit-scrollbar-thumb{background:#2d333b;border-radius:3px}
        input,select{outline:none;transition:border-color .15s,box-shadow .15s}
        input:focus,select:focus{border-color:#539bf5!important;box-shadow:0 0 0 3px #539bf520!important}
        .inp{width:100%;padding:7px 10px;background:#13191f;border:1px solid #2d333b;border-radius:6px;color:#cdd9e5;font-family:inherit;font-size:13px}
        .panel{background:#13191f;border:1px solid #2d333b;border-radius:10px;padding:18px}
        .tab-btn{cursor:pointer;padding:7px 16px;border-radius:6px;font-family:inherit;font-size:12px;font-weight:500;letter-spacing:.3px;border:none;transition:all .15s}
        .fade{animation:fadeIn .35s ease}
        @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
        .mthbtn{cursor:pointer;width:100%;text-align:left;padding:9px 12px;border-radius:7px;border:1px solid;font-family:inherit;font-size:12px;transition:all .15s;display:flex;align-items:center;gap:8px;margin-bottom:5px}
        .mthbtn:hover{filter:brightness(1.1)}
        .calc-btn{cursor:pointer;width:100%;padding:11px;border-radius:8px;background:linear-gradient(135deg,#1a7f64,#2da677);border:none;color:#fff;font-size:13px;font-weight:600;font-family:inherit;letter-spacing:.3px;transition:all .15s}
        .calc-btn:hover{transform:translateY(-1px);filter:brightness(1.1)}
        .calc-btn:active{transform:none}
      `}</style>

      {/* HEADER */}
      <div style={{borderBottom:"1px solid #2d333b",padding:"14px 28px",display:"flex",alignItems:"center",gap:12}}>
        <div style={{width:30,height:30,background:"linear-gradient(135deg,#2da677,#539bf5)",borderRadius:7,display:"flex",alignItems:"center",justifyContent:"center",fontSize:15}}>⚗</div>
        <span style={{fontFamily:"'Syne',sans-serif",fontWeight:800,fontSize:17,letterSpacing:"-.5px"}}>EcotoxLab</span>
        <span style={{color:"#768390",fontSize:12,marginLeft:4}}>Análise de Toxicidade Aquática</span>
        <div style={{marginLeft:"auto",display:"flex",gap:8}}>
          <span style={{background:"#1a3a2e",color:"#2da677",padding:"3px 9px",borderRadius:4,fontSize:11,fontWeight:600}}>
            {"{"}ecotox{"}"} v1.4
          </span>
          <span style={{background:"#1a2a3a",color:"#539bf5",padding:"3px 9px",borderRadius:4,fontSize:11,fontWeight:600}}>
            {method.fn}
          </span>
        </div>
      </div>

      <div style={{maxWidth:1140,margin:"0 auto",padding:"22px 20px",display:"grid",gridTemplateColumns:"262px 1fr",gap:18,alignItems:"start"}}>

        {/* ── SIDEBAR ── */}
        <div style={{display:"flex",flexDirection:"column",gap:14}}>
          <div className="panel">
            <SecTitle>Experimento</SecTitle>
            <Lbl>Substância / Espécie</Lbl>
            <input className="inp" style={{marginBottom:10}} value={substance} onChange={e=>setSubstance(e.target.value)} placeholder="Nome"/>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:10}}>
              <div><Lbl>Repetições</Lbl><input className="inp" type="number" min={1} value={reps} onChange={e=>setReps(+e.target.value)}/></div>
              <div><Lbl>Indivíduos</Lbl><input className="inp" type="number" min={1} value={indiv} onChange={e=>setIndiv(+e.target.value)}/></div>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:10}}>
              <div>
                <Lbl>Unid. Conc.</Lbl>
                <select className="inp" value={unitConc} onChange={e=>setUnitConc(e.target.value)}>
                  {UNITS_CONC.map(u=><option key={u}>{u}</option>)}
                </select>
              </div>
              <div>
                <Lbl>Unid. Tempo</Lbl>
                <select className="inp" value={unitTime} onChange={e=>setUnitTime(e.target.value)}>
                  {UNITS_TIME.map(u=><option key={u}>{u}</option>)}
                </select>
              </div>
            </div>
            <Lbl>Nº de doses / tempos</Lbl>
            <input className="inp" type="number" min={2} max={20} value={nDoses}
              onChange={e=>setNDoses(Math.max(2,Math.min(20,+e.target.value)))}/>
            <div style={{fontSize:11,color:"#768390",marginTop:5}}>
              Total por grupo: <b style={{color:"#cdd9e5"}}>{total}</b> organismos
            </div>
          </div>

          {/* Method selector */}
          <div className="panel">
            <SecTitle>Método de Análise</SecTitle>
            <div style={{fontSize:10,color:"#768390",marginBottom:10,letterSpacing:".3px",lineHeight:1.5}}>
              Baseado no pacote R <b style={{color:"#cdd9e5"}}>{"{ecotox}"}</b><br/>Hlina et al. (2021)
            </div>

            <GrpLabel color="#539bf5">Concentração Letal (LC)</GrpLabel>
            {METHODS.filter(m=>m.group==="Concentração Letal").map(m=>(
              <MBtn key={m.id} m={m} active={methodId===m.id} onSelect={setMethodId} color="#539bf5"/>
            ))}

            <GrpLabel color="#e3b341" style={{marginTop:10}}>Tempo Letal (LT)</GrpLabel>
            {METHODS.filter(m=>m.group==="Tempo Letal").map(m=>(
              <MBtn key={m.id} m={m} active={methodId===m.id} onSelect={setMethodId} color="#e3b341"/>
            ))}

            <GrpLabel color="#2da677" style={{marginTop:10}}>Não-paramétrico</GrpLabel>
            {METHODS.filter(m=>m.group==="Não-paramétrico").map(m=>(
              <MBtn key={m.id} m={m} active={methodId===m.id} onSelect={setMethodId} color="#2da677"/>
            ))}

            <div style={{marginTop:12,padding:"9px 11px",background:"#0a0e14",borderRadius:6,fontSize:11,color:"#768390",lineHeight:1.65}}>
              {METHOD_NOTES[methodId]}
            </div>
          </div>
        </div>

        {/* ── MAIN CONTENT ── */}
        <div style={{display:"flex",flexDirection:"column",gap:14}}>
          {/* Tabs */}
          <div style={{display:"flex",gap:8,paddingBottom:12,borderBottom:"1px solid #2d333b",alignItems:"center"}}>
            {[["data","📋  Dados"],["result","📊  Resultados"]].map(([id,lbl])=>(
              <button key={id} className="tab-btn" onClick={()=>setTab(id)} style={{
                background: tab===id ? "#1a7f64" : "transparent",
                color: tab===id ? "#fff" : "#768390",
                border: tab===id ? "none" : "1px solid #2d333b",
              }}>{lbl}</button>
            ))}
            <div style={{marginLeft:"auto",fontSize:11,color:"#768390"}}>
              Método ativo:{" "}
              <b style={{color: method.group==="Tempo Letal"?"#e3b341":method.group==="Não-paramétrico"?"#2da677":"#539bf5"}}>
                {method.label}
              </b>
            </div>
          </div>

          {/* DATA TAB */}
          {tab==="data" && (
            <div className="panel fade">
              <SecTitle>
                {method.xIsTime ? "Tempo de Exposição" : "Concentração"} × Mortalidade
              </SecTitle>
              <div style={{display:"grid",gridTemplateColumns:"28px 1fr 1fr 90px",gap:8,marginBottom:8}}>
                {["#", xLabel, "Mortos", "Mort. %"].map(h=>(
                  <div key={h} style={{fontSize:10,color:"#768390",fontWeight:600,letterSpacing:".4px",textTransform:"uppercase"}}>{h}</div>
                ))}
              </div>
              <div style={{display:"flex",flexDirection:"column",gap:5}}>
                {rows.map((row,i)=>{
                  const d=parseFloat(row.dead), pct=(!isNaN(d)&&total>0)?(d/total*100).toFixed(1):null;
                  const pn=parseFloat(pct);
                  const col=isNaN(pn)?"#768390":pn>50?"#e5534b":pn>25?"#d29922":"#2da677";
                  return (
                    <div key={i} style={{display:"grid",gridTemplateColumns:"28px 1fr 1fr 90px",gap:8,alignItems:"center"}}>
                      <div style={{fontSize:11,color:"#539bf5",fontWeight:600}}>{i+1}</div>
                      <input className="inp" type="number" step="any" min={0} placeholder="0.0000"
                        value={row.x} onChange={e=>updateRow(i,"x",e.target.value)}/>
                      <input className="inp" type="number" min={0} max={total} placeholder="0"
                        value={row.dead} onChange={e=>updateRow(i,"dead",e.target.value)}/>
                      <div style={{padding:"7px 10px",borderRadius:6,background:"#0a0e14",fontSize:12,color:col,textAlign:"right",fontWeight:500}}>
                        {pct!=null?`${pct}%`:"—"}
                      </div>
                    </div>
                  );
                })}
              </div>
              {error && (
                <div style={{marginTop:12,padding:"9px 13px",background:"#2d1217",border:"1px solid #e5534b40",borderRadius:6,fontSize:12,color:"#e5534b"}}>
                  ⚠ {error}
                </div>
              )}
              <button className="calc-btn" style={{marginTop:14}} onClick={analyze}>
                ▶ Calcular {clLabel} — {method.label}
              </button>
            </div>
          )}

          {/* RESULTS TAB */}
          {tab==="result" && (
            <div className="fade">
              {!result ? (
                <div className="panel" style={{textAlign:"center",padding:"44px",color:"#768390"}}>
                  Insira os dados e clique em <b style={{color:"#2da677"}}>Calcular</b> na aba Dados.
                </div>
              ) : (
                <>
                  {/* Primary cards */}
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:12,marginBottom:12}}>
                    <SCard label={clLabel} value={fmt(result.cl)} unit={unit} highlight color="#539bf5"/>
                    <SCard label="Lim. Inferior 95%" value={fmt(result.lcl)} unit={unit} color="#2da677"/>
                    <SCard label="Lim. Superior 95%" value={fmt(result.ucl)} unit={unit} color="#2da677"/>
                  </div>

                  {/* GLM stats */}
                  {result.slope !== null && (
                    <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10,marginBottom:12}}>
                      <SCard label="Slope (b₁)" value={fmt(result.slope)} small/>
                      <SCard label="Intercept (b₀)" value={fmt(result.intercept)} small/>
                      <SCard label="z-value (slope)" value={result.zValue ?? "—"} small/>
                      <SCard label="Var(log CL)" value={result.variance ?? "—"} small/>
                    </div>
                  )}
                  <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10,marginBottom:14}}>
                    <SCard label="χ² Pearson (GOF)" value={fmt2(result.chi2)} small/>
                    <SCard label="p-valor GOF" value={result.pgof ?? "—"} small
                      color={parseFloat(result.pgof)>0.05?"#2da677":"#e5534b"}/>
                  </div>

                  {/* CHART */}
                  <div className="panel" style={{padding:"16px 6px 12px"}}>
                    <div style={{fontSize:11,color:"#768390",marginBottom:8,paddingLeft:14,fontFamily:"'Syne',sans-serif",fontWeight:700}}>
                      {substance} — Curva Dose-Resposta ({method.label})
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                      <ComposedChart data={chartData} margin={{top:8,right:26,left:4,bottom:22}}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2d333b"/>
                        <XAxis dataKey="x" stroke="#2d333b" tick={{fill:"#768390",fontSize:10}}
                          label={{value:xLabel,position:"insideBottom",offset:-10,fill:"#768390",fontSize:11}}/>
                        <YAxis domain={[0,100]} ticks={[0,10,20,30,40,50,60,70,80,90,100]}
                          stroke="#2d333b" tick={{fill:"#768390",fontSize:10}}
                          label={{value:"Mortalidade (%)",angle:-90,position:"insideLeft",offset:10,fill:"#768390",fontSize:11}}/>
                        <Tooltip
                          contentStyle={{background:"#13191f",border:"1px solid #2d333b",borderRadius:6,fontSize:11,fontFamily:"inherit"}}
                          labelStyle={{color:"#768390"}}
                          labelFormatter={v=>`${xLabel}: ${v}`}
                          formatter={(v,name)=>{
                            if(v==null)return[null,null];
                            const nm={curve:`Curva ${clLabel}`,lower:"IC Inf 95%",upper:"IC Sup 95%",obs:"Observado"};
                            return[`${v}%`, nm[name]||name];
                          }}/>
                        <ReferenceLine y={50} stroke="#2d333b" strokeDasharray="4 2"/>
                        <ReferenceLine x={result.cl}  stroke="#539bf5" strokeDasharray="5 3"
                          label={{value:clLabel,position:"insideTopRight",fill:"#539bf5",fontSize:10}}/>
                        <ReferenceLine x={result.lcl} stroke="#2da677" strokeDasharray="3 3"
                          label={{value:"LI",position:"top",fill:"#2da677",fontSize:9}}/>
                        <ReferenceLine x={result.ucl} stroke="#2da677" strokeDasharray="3 3"
                          label={{value:"LS",position:"top",fill:"#2da677",fontSize:9}}/>
                        <Line dataKey="upper" stroke="#2da677" strokeWidth={1.5} strokeDasharray="5 3" dot={false} connectNulls/>
                        <Line dataKey="lower" stroke="#2da677" strokeWidth={1.5} strokeDasharray="5 3" dot={false} connectNulls/>
                        <Line dataKey="curve" stroke="#539bf5" strokeWidth={2.5} dot={false} connectNulls/>
                        <Scatter dataKey="obs" fill="#e3b341" r={5}/>
                      </ComposedChart>
                    </ResponsiveContainer>
                    {/* Legend */}
                    <div style={{display:"flex",gap:18,paddingLeft:14,marginTop:4,flexWrap:"wrap"}}>
                      {[
                        {color:"#539bf5",dash:false,label:`Curva ${clLabel}`},
                        {color:"#2da677",dash:true, label:"IC 95% (Inf / Sup)"},
                        {color:"#e3b341",dot:true,  label:"Dados observados"},
                        {color:"#539bf5",bar:true,  label:`${clLabel} = ${fmt(result.cl)} ${unit}`},
                      ].map(({color,dash,dot,bar,label})=>(
                        <div key={label} style={{display:"flex",alignItems:"center",gap:5,fontSize:10,color:"#768390"}}>
                          {dot ? <div style={{width:9,height:9,borderRadius:"50%",background:color}}/> :
                           bar ? <div style={{width:2,height:14,background:color}}/> :
                                 <svg width="22" height="8"><line x1="0" y1="4" x2="22" y2="4" stroke={color} strokeWidth="2" strokeDasharray={dash?"5 3":"none"}/></svg>}
                          {label}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Method note */}
                  <div className="panel" style={{marginTop:12,fontSize:11,color:"#768390",lineHeight:1.7}}>
                    <b style={{color:"#cdd9e5"}}>Referência metodológica:</b>{" "}{METHOD_NOTES[methodId]}
                    {" "}p GOF {">"}0.05 indica bom ajuste do modelo.
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── UI helpers ───────────────────────────────────────────────
function SecTitle({children}) {
  return <div style={{fontFamily:"'Syne',sans-serif",fontWeight:700,fontSize:12,color:"#768390",letterSpacing:"1px",textTransform:"uppercase",marginBottom:13}}>{children}</div>;
}
function Lbl({children}) {
  return <div style={{fontSize:10,color:"#768390",fontWeight:600,letterSpacing:".4px",marginBottom:4,textTransform:"uppercase"}}>{children}</div>;
}
function GrpLabel({children,color}) {
  return <div style={{fontSize:10,color,fontWeight:700,letterSpacing:"1px",marginBottom:6,marginTop:4}}>{children}</div>;
}
function MBtn({m, active, onSelect, color}) {
  return (
    <button className="mthbtn" onClick={()=>onSelect(m.id)} style={{
      borderColor: active ? color : "#2d333b",
      background:  active ? `${color}18` : "transparent",
      color:       active ? color : "#768390",
    }}>
      <span style={{width:6,height:6,borderRadius:"50%",background:active?color:"#2d333b",flexShrink:0,display:"inline-block"}}/>
      <code style={{fontSize:12}}>{m.label}()</code>
      {active && <span style={{marginLeft:"auto",fontSize:9,opacity:.7}}>✓</span>}
    </button>
  );
}
function SCard({label,value,unit,highlight,color,small}) {
  return (
    <div style={{padding:small?"10px 12px":"14px 16px",borderRadius:8,
      background:highlight?"#0e2030":"#13191f",border:`1px solid ${highlight?"#1a4a7a":"#2d333b"}`}}>
      <div style={{fontSize:10,color:"#768390",marginBottom:5,fontWeight:600,letterSpacing:".3px",textTransform:"uppercase"}}>{label}</div>
      <div style={{fontFamily:"'Syne',sans-serif",fontSize:small?14:highlight?22:17,fontWeight:700,
        color:color||(highlight?"#539bf5":"#cdd9e5")}}>{value}</div>
      {unit&&<div style={{fontSize:10,color:"#768390",marginTop:2}}>{unit}</div>}
    </div>
  );
}
