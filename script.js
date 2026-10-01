/* ============================================================
   TRADEPULSE v3.0 — script.js
   Groww API backend integration
   Features: Live prices, Buy/Sell orders, Top Gainers/Losers
   ============================================================ */

// ── Backend URL (your Flask server) ──────────────────────────
const BACKEND = "http://localhost:5000/api";

// ── AUTH CHECK ───────────────────────────────────────────────
(function authCheck() {
  const user = JSON.parse(localStorage.getItem('tp_loggedin') || 'null');
  if (!user) { window.location.href = 'login.html'; return; }
  document.getElementById('userName').textContent = user.name.split(' ')[0];
  const av = document.getElementById('userAvatar');
  if (av) av.textContent = user.name.charAt(0).toUpperCase();
})();

function logout() {
  localStorage.removeItem('tp_loggedin');
  window.location.href = 'login.html';
}

// ── STOCK DATABASE (fallback if backend offline) ──────────────
const BASE_STOCKS = {
  RELIANCE:   { price:2847.35, change:+1.2,  high:2880,  low:2810,  w52:3024,  vol:'4.2M', rsi:58, macd:'BULLISH', signal:'BUY',  entry:2820,  stop:2790,  target:2920  },
  TCS:        { price:3612.10, change:-0.4,  high:3640,  low:3590,  w52:4255,  vol:'1.1M', rsi:44, macd:'BEARISH', signal:'SELL', entry:3640,  stop:3665,  target:3520  },
  INFY:       { price:1487.60, change:+0.8,  high:1510,  low:1472,  w52:1930,  vol:'3.5M', rsi:51, macd:'NEUTRAL', signal:'HOLD', entry:1478,  stop:1460,  target:1530  },
  HDFCBANK:   { price:1648.90, change:+1.5,  high:1672,  low:1630,  w52:1794,  vol:'2.8M', rsi:62, macd:'BULLISH', signal:'BUY',  entry:1638,  stop:1615,  target:1695  },
  WIPRO:      { price:452.75,  change:-1.1,  high:462,   low:448,   w52:577,   vol:'5.1M', rsi:39, macd:'BEARISH', signal:'SELL', entry:462,   stop:472,   target:435   },
  TATAMOTORS: { price:967.40,  change:+2.3,  high:985,   low:952,   w52:1064,  vol:'8.7M', rsi:67, macd:'BULLISH', signal:'BUY',  entry:955,   stop:938,   target:1005  },
  SBIN:       { price:812.55,  change:+0.6,  high:825,   low:805,   w52:912,   vol:'12M',  rsi:55, macd:'BULLISH', signal:'BUY',  entry:808,   stop:795,   target:845   },
  BAJFINANCE: { price:7124.30, change:-0.9,  high:7200,  low:7080,  w52:8192,  vol:'0.9M', rsi:42, macd:'BEARISH', signal:'SELL', entry:7200,  stop:7260,  target:6980  },
  ITC:        { price:467.80,  change:+0.3,  high:472,   low:463,   w52:525,   vol:'15M',  rsi:53, macd:'NEUTRAL', signal:'HOLD', entry:465,   stop:458,   target:482   },
  LT:         { price:3542.20, change:+1.1,  high:3580,  low:3510,  w52:3895,  vol:'1.8M', rsi:60, macd:'BULLISH', signal:'BUY',  entry:3530,  stop:3495,  target:3620  },
  AXISBANK:   { price:1124.45, change:-0.7,  high:1140,  low:1115,  w52:1235,  vol:'4.5M', rsi:41, macd:'BEARISH', signal:'SELL', entry:1140,  stop:1155,  target:1095  },
  MARUTI:     { price:11240.0, change:+0.9,  high:11350, low:11180, w52:12200, vol:'0.5M', rsi:57, macd:'BULLISH', signal:'BUY',  entry:11200, stop:11050, target:11500 },
};

const LIVE = {};
Object.keys(BASE_STOCKS).forEach(s => { LIVE[s] = BASE_STOCKS[s].price; });

const INDICES_DATA = [
  { id:'sensex',    name:'BSE SENSEX',  base:74520.30, change:+0.42 },
  { id:'nifty',     name:'NIFTY 50',    base:22608.75, change:+0.38 },
  { id:'banknifty', name:'BANK NIFTY',  base:48214.60, change:+0.71 },
];

let backendOnline = false;

// ── UTILS ─────────────────────────────────────────────────────
const fmt  = n => Number(n).toLocaleString('en-IN', { minimumFractionDigits:2, maximumFractionDigits:2 });
const fmtK = n => Number(n).toLocaleString('en-IN', { maximumFractionDigits:0 });
const rand = (b,p) => b*(1+(Math.random()-.5)*p/100);
function fmtVol(v) {
  if (v >= 10000000) return (v/10000000).toFixed(1)+'Cr';
  if (v >= 100000)   return (v/100000).toFixed(1)+'L';
  if (v >= 1000)     return (v/1000).toFixed(1)+'K';
  return String(v);
}
function genLine(pts, base, trend, noise) {
  let v = base;
  return Array.from({length:pts}, () => {
    v = +(v*(1+trend+(Math.random()-.49)*noise)).toFixed(2);
    return v;
  });
}

// ── CHECK BACKEND STATUS ──────────────────────────────────────
async function checkBackend() {
  try {
    const res  = await fetch(`${BACKEND}/status`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    backendOnline = true;
    showBackendStatus(true, data.groww_api);
  } catch {
    backendOnline = false;
    showBackendStatus(false, "");
  }
}

function showBackendStatus(online, growwStatus) {
  const dot = document.querySelector('.live-dot');
  const tag = document.querySelector('.live-tag');
  if (!dot || !tag) return;
  if (online && growwStatus.includes('connected')) {
    dot.style.background = '#00e676';
    dot.style.boxShadow  = '0 0 8px #00e676';
    tag.textContent = 'GROWW LIVE';
    tag.style.color = '#00e676';
  } else if (online) {
    dot.style.background = '#ffb800';
    dot.style.boxShadow  = '0 0 8px #ffb800';
    tag.textContent = 'BACKEND ON';
    tag.style.color = '#ffb800';
  } else {
    dot.style.background = '#ff4b4b';
    dot.style.boxShadow  = '0 0 8px #ff4b4b';
    tag.textContent = 'OFFLINE';
    tag.style.color = '#ff4b4b';
  }
}

// ── FETCH ALL PRICES FROM BACKEND (Yahoo Finance LIVE) ────────
async function fetchAllPrices() {
  backendOnline = true; // always try
  // Always try to fetch — don't depend on backendOnline flag
  backendOnline = true;
  try {
    const res  = await fetch(`${BACKEND}/live/all`, { signal: AbortSignal.timeout(15000) });
    const data = await res.json();
    Object.keys(data).forEach(sym => {
      const d = data[sym];
      if (!d || !d.price) return;
      const p = parseFloat(d.price);
      if (!p || p <= 0) return;
      LIVE[sym] = p;
      if (!BASE_STOCKS[sym]) BASE_STOCKS[sym] = {};
      BASE_STOCKS[sym].price  = p;
      BASE_STOCKS[sym].change = parseFloat(d.change)  || 0;
      BASE_STOCKS[sym].high   = parseFloat(d.high)    || p;
      BASE_STOCKS[sym].low    = parseFloat(d.low)     || p;
      BASE_STOCKS[sym].w52    = parseFloat(d.w52high) || p;
      BASE_STOCKS[sym].vol    = fmtVol(parseInt(d.volume) || 0);
      BASE_STOCKS[sym].signal = d.signal || 'HOLD';
      BASE_STOCKS[sym].rsi    = d.rsi    || 50;
      BASE_STOCKS[sym].macd   = d.macd   || 'NEUTRAL';
      BASE_STOCKS[sym].entry  = parseFloat(d.entry)  || p;
      BASE_STOCKS[sym].stop   = parseFloat(d.stop)   || p;
      BASE_STOCKS[sym].target = parseFloat(d.target) || p;
    });
    console.log('✅ Real prices applied:', Object.keys(LIVE).map(s=>s+':'+LIVE[s]).join(', '));
    buildTicker();
    renderWatchlist();
    renderMovers();
    updateLiveSignalPrice();
    updatePortfolioPrices();
  } catch (e) {
    console.warn('fetchAllPrices failed:', e.message);
    tickPrices();
  }
}

// ── FETCH SINGLE STOCK LIVE PRICE (Yahoo Finance) ─────────────
async function fetchLivePrice(sym) {
  try {
    const res  = await fetch(`${BACKEND}/live/${sym}`, { signal: AbortSignal.timeout(8000) });
    const data = await res.json();
    if (data.price) {
      LIVE[sym] = data.price;
      if (BASE_STOCKS[sym]) {
        BASE_STOCKS[sym].price  = data.price;
        BASE_STOCKS[sym].change = data.change  || BASE_STOCKS[sym].change;
        BASE_STOCKS[sym].high   = data.high    || BASE_STOCKS[sym].high;
        BASE_STOCKS[sym].low    = data.low     || BASE_STOCKS[sym].low;
        BASE_STOCKS[sym].signal = data.signal  || BASE_STOCKS[sym].signal;
        BASE_STOCKS[sym].rsi    = data.rsi     || BASE_STOCKS[sym].rsi;
        BASE_STOCKS[sym].macd   = data.macd    || BASE_STOCKS[sym].macd;
        BASE_STOCKS[sym].entry  = data.entry   || BASE_STOCKS[sym].entry;
        BASE_STOCKS[sym].stop   = data.stop    || BASE_STOCKS[sym].stop;
        BASE_STOCKS[sym].target = data.target  || BASE_STOCKS[sym].target;
      }
    }
    return data;
  } catch(e) {
    return null;
  }
}

// ── FETCH INDICES ─────────────────────────────────────────────
async function fetchIndices() {
  try {
    const res  = await fetch(`${BACKEND}/indices`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    ['sensex','nifty','banknifty'].forEach(key => {
      const d  = data[key];
      const el = document.getElementById(`iv-${key}`);
      const ce = document.getElementById(`ic-${key}`);
      if (el) el.textContent = fmt(d.value);
      if (ce) {
        ce.textContent = `${d.change>=0?'▲':'▼'} ${Math.abs(d.change)}%`;
        ce.className   = `idx-chg ${d.change>=0?'up':'dn'}`;
      }
      const found = INDICES_DATA.find(i=>i.id===key);
      if (found) { found.base = d.value; found.change = d.change; }
    });
  } catch {}
}

// ── FETCH TOP GAINERS ─────────────────────────────────────────
async function fetchGainers() {
  try {
    const res  = await fetch(`${BACKEND}/gainers`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    renderGainersLosers('gainers', data, true);
  } catch {
    // fallback — sort BASE_STOCKS
    const sorted = Object.entries(BASE_STOCKS)
      .map(([sym,d])=>({ symbol:sym, price:d.price, change_pct:d.change }))
      .filter(s=>s.change_pct>0).sort((a,b)=>b.change_pct-a.change_pct).slice(0,5);
    renderGainersLosers('gainers', sorted, true);
  }
}

// ── FETCH TOP LOSERS ──────────────────────────────────────────
async function fetchLosers() {
  try {
    const res  = await fetch(`${BACKEND}/losers`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    renderGainersLosers('losers', data, false);
  } catch {
    const sorted = Object.entries(BASE_STOCKS)
      .map(([sym,d])=>({ symbol:sym, price:d.price, change_pct:d.change }))
      .filter(s=>s.change_pct<0).sort((a,b)=>a.change_pct-b.change_pct).slice(0,5);
    renderGainersLosers('losers', sorted, false);
  }
}

// ── RENDER GAINERS / LOSERS ───────────────────────────────────
function renderGainersLosers(containerId, data, isGainer) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = data.map(s => `
    <div class="mover-item" onclick="loadStock('${s.symbol}')">
      <div>
        <div class="mover-sym">${s.symbol}</div>
        <div style="font-family:var(--font-mono);font-size:10px;color:var(--muted)">₹${fmt(s.price)}</div>
      </div>
      <span class="mover-chg ${isGainer?'up':'dn'}">
        ${isGainer?'▲':' ▼'} ${isGainer?'+':''}${s.change_pct}%
      </span>
    </div>`).join('');
}

// ── SIMULATION ENGINE (fallback) ──────────────────────────────
function tickPrices() {
  Object.keys(LIVE).forEach(sym => {
    const base  = BASE_STOCKS[sym]?.price || 1000;
    const trend = (BASE_STOCKS[sym]?.change || 0) >= 0 ? 0.0001 : -0.0001;
    const noise = (Math.random()-.49)*0.001;
    // Only simulate small tick movement around current live price
    LIVE[sym] = +(LIVE[sym]*(1+trend+noise)).toFixed(2);
    // Keep within 1% of base (not 5%) so it stays close to real price
    if (LIVE[sym] > base*1.01) LIVE[sym] = +(base*1.01).toFixed(2);
    if (LIVE[sym] < base*0.99) LIVE[sym] = +(base*0.99).toFixed(2);
  });
  checkAlerts();
}

// ── CLOCK ─────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  const t   = [now.getHours(), now.getMinutes(), now.getSeconds()].map(x=>String(x).padStart(2,'0')).join(':');
  document.getElementById('navClock').textContent = t;
  const h = now.getHours(), m = now.getMinutes();
  const open = (h>9||(h===9&&m>=15))&&(h<15||(h===15&&m<=30));
  const ms = document.getElementById('mktStatus');
  if (ms) { ms.textContent=open?'● MARKET OPEN':'● MARKET CLOSED'; ms.style.color=open?'var(--green)':'var(--red)'; }
}
setInterval(updateClock, 1000);
updateClock();

// ── TICKER ────────────────────────────────────────────────────
function buildTicker() {
  const items = [...Object.entries(BASE_STOCKS),...Object.entries(BASE_STOCKS)];
  document.getElementById('tickerTrack').innerHTML = items.map(([sym,d]) => {
    const cls=d.change>=0?'up':'dn', arr=d.change>=0?'▲':'▼';
    return `<div class="tick-item"><span class="tick-sym">${sym}</span><span class="tick-val" id="tk-${sym}">₹${fmt(d.price)}</span><span class="tick-chg ${cls}">${arr}${Math.abs(d.change)}%</span></div>`;
  }).join('');
}
function updateTicker() {
  Object.keys(LIVE).forEach(sym => {
    const el = document.getElementById(`tk-${sym}`);
    if (el) el.textContent = '₹'+fmt(LIVE[sym]);
  });
}

// ── INDICES ───────────────────────────────────────────────────
let idxCharts = {};
function renderIndices() {
  document.getElementById('indicesRow').innerHTML = INDICES_DATA.map(idx => `
    <div class="idx-card">
      <div class="idx-label">${idx.name}</div>
      <div class="idx-val" id="iv-${idx.id}">${fmt(idx.base)}</div>
      <div class="idx-chg ${idx.change>=0?'up':'dn'}" id="ic-${idx.id}">
        ${idx.change>=0?'▲':'▼'} ${Math.abs(idx.change)}%
      </div>
      <div class="idx-mini-chart"><canvas id="sp-${idx.id}"></canvas></div>
    </div>`).join('');
  INDICES_DATA.forEach(idx => {
    const ctx = document.getElementById(`sp-${idx.id}`).getContext('2d');
    const col = idx.change>=0?'#00e676':'#ff4b4b';
    const data = genLine(30,idx.base,idx.change>=0?0.0003:-0.0003,0.003);
    if (idxCharts[idx.id]) idxCharts[idx.id].destroy();
    idxCharts[idx.id] = new Chart(ctx,{type:'line',data:{labels:data.map((_,i)=>i),datasets:[{data,borderColor:col,borderWidth:1.5,pointRadius:0,tension:0.4,fill:{target:'origin',above:col+'18'}}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{display:false},y:{display:false}},animation:{duration:0}}});
  });
}

// ── QUICK PICKS ───────────────────────────────────────────────
function buildQuickPicks() {
  const picks = ['RELIANCE','TCS','INFY','SBIN','TATAMOTORS','HDFCBANK'];
  document.getElementById('quickPicks').innerHTML = picks.map(s=>
    `<span class="quick-chip" onclick="loadStock('${s}')">${s}</span>`).join('');
}

// ── MAIN CHART ────────────────────────────────────────────────
let mainChartInst=null, chartMode='5m', chartSym='', chartData=[];
function buildMainChart(sym) {
  chartSym = sym;
  const ctx = document.getElementById('mainChart').getContext('2d');
  if (mainChartInst) mainChartInst.destroy();
  const pts = chartMode==='1m'?60:chartMode==='5m'?48:chartMode==='15m'?30:18;
  const d   = BASE_STOCKS[sym]||{price:1000,change:0.5};
  const ms2 = chartMode==='1m'?1:chartMode==='5m'?5:chartMode==='15m'?15:60;
  const col = d.change>=0?'#00e676':'#ff4b4b';
  chartData = genLine(pts,d.price*0.995,d.change>=0?0.0002:-0.0002,0.002);
  const labels = chartData.map((_,i)=>{ const t=9*60+15+i*ms2; return `${Math.floor(t/60)}:${String(t%60).padStart(2,'0')}`; });
  document.getElementById('chartTitle').textContent = sym;
  mainChartInst = new Chart(ctx,{type:'line',data:{labels,datasets:[{label:sym,data:[...chartData],borderColor:col,borderWidth:1.5,pointRadius:0,pointHoverRadius:4,tension:0.3,fill:{target:'origin',above:col+'14'}}]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:300},plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`₹${fmt(c.raw)}`}}},scales:{x:{ticks:{color:'#5a7060',font:{family:'Share Tech Mono',size:10}},grid:{color:'rgba(0,200,120,0.05)'}},y:{ticks:{color:'#5a7060',font:{family:'Share Tech Mono',size:10},callback:v=>'₹'+fmt(v)},grid:{color:'rgba(0,200,120,0.05)'}}}}}); 
}
function liveUpdateChart() {
  if (!chartSym||!mainChartInst) return;
  chartData.push(LIVE[chartSym]||chartData[chartData.length-1]);
  chartData.shift();
  mainChartInst.data.datasets[0].data=[...chartData];
  mainChartInst.update('none');
}
function setChartMode(m) { chartMode=m; if(chartSym) buildMainChart(chartSym); }

// ── MARKET MOVERS ─────────────────────────────────────────────
function renderMovers() {
  const all     = Object.entries(BASE_STOCKS).map(([sym,d])=>({sym,chg:d.change}));
  const gainers = all.filter(s=>s.chg>0).sort((a,b)=>b.chg-a.chg).slice(0,4);
  const losers  = all.filter(s=>s.chg<0).sort((a,b)=>a.chg-b.chg).slice(0,4);
  const gEl = document.getElementById('gainers');
  const lEl = document.getElementById('losers');
  if (gEl) gEl.innerHTML = gainers.map(s=>`<div class="mover-item" onclick="loadStock('${s.sym}')"><div><div class="mover-sym">${s.sym}</div><div style="font-family:var(--font-mono);font-size:10px;color:var(--muted)">₹${fmt(LIVE[s.sym]||0)}</div></div><span class="mover-chg up">▲ +${s.chg}%</span></div>`).join('');
  if (lEl) lEl.innerHTML = losers.map(s=>`<div class="mover-item" onclick="loadStock('${s.sym}')"><div><div class="mover-sym">${s.sym}</div><div style="font-family:var(--font-mono);font-size:10px;color:var(--muted)">₹${fmt(LIVE[s.sym]||0)}</div></div><span class="mover-chg dn">▼ ${s.chg}%</span></div>`).join('');
}

// ── WATCHLIST ─────────────────────────────────────────────────
const WATCHLIST_SYMS = ['RELIANCE','TCS','INFY','HDFCBANK','SBIN','TATAMOTORS','ITC','LT'];
function renderWatchlist() {
  document.getElementById('watchlistBody').innerHTML = WATCHLIST_SYMS.map(sym => {
    const d=BASE_STOCKS[sym], ltp=LIVE[sym]||d.price, cls=d.change>=0?'green':'red';
    return `<tr style="cursor:pointer" onclick="loadStock('${sym}')">
      <td style="color:var(--text);letter-spacing:1px;font-weight:500">${sym}</td>
      <td class="${cls}" id="wl-${sym}">₹${fmt(ltp)}</td>
      <td class="${cls}">${d.change>=0?'+':''}${d.change}%</td>
      <td class="green">₹${fmt(d.high)}</td>
      <td class="red">₹${fmt(d.low)}</td>
      <td style="color:var(--muted)">${d.vol}</td>
      <td><span class="badge ${d.signal.toLowerCase()}">${d.signal}</span></td>
    </tr>`;
  }).join('');
}
function updateWatchlistPrices() {
  WATCHLIST_SYMS.forEach(sym => {
    const el = document.getElementById(`wl-${sym}`);
    if (el&&LIVE[sym]) el.textContent = '₹'+fmt(LIVE[sym]);
  });
}

// ── ORDER BOOK ────────────────────────────────────────────────
function renderOrderBook(sym, price, buyDepth, sellDepth) {
  document.getElementById('orderBookPanel').style.display = 'block';
  document.getElementById('obSymbol').textContent = sym;
  updateOrderBook(price, buyDepth, sellDepth);
}
function updateOrderBook(price, buyDepth, sellDepth) {
  if (!price) return;
  let html = '';
  for (let i=5;i>=1;i--) {
    const bd = buyDepth?.[i-1]  || { price: +(price-i*(price*0.0005)).toFixed(2), quantity: Math.round(Math.random()*800+200) };
    const sd = sellDepth?.[i-1] || { price: +(price+i*(price*0.0005)).toFixed(2), quantity: Math.round(Math.random()*800+200) };
    html += `<tr><td class="ob-qty">${fmtK(bd.quantity)}</td><td class="ob-bid">₹${fmt(bd.price)}</td><td class="ob-ask">₹${fmt(sd.price)}</td><td class="ob-qty">${fmtK(sd.quantity)}</td></tr>`;
  }
  document.getElementById('obBody').innerHTML = html;
  const bidPct = Math.round(45+Math.random()*20), askPct=100-bidPct;
  document.getElementById('obBidBar').style.width = bidPct+'%';
  document.getElementById('obAskBar').style.width = askPct+'%';
  document.getElementById('obBidPct').textContent = bidPct+'% BUY';
  document.getElementById('obAskPct').textContent = askPct+'% SELL';
}

// ── BUY / SELL ORDER MODAL ────────────────────────────────────
let currentOrderSym = '', currentOrderPrice = 0;

function openBuyModal(sym, price) {
  currentOrderSym   = sym || document.getElementById('sName').textContent;
  currentOrderPrice = price || LIVE[currentOrderSym] || 0;
  document.getElementById('orderSymbol').textContent  = currentOrderSym;
  document.getElementById('orderPrice').textContent   = '₹'+fmt(currentOrderPrice);
  document.getElementById('orderQty').value           = '1';
  document.getElementById('orderTotal').textContent   = '₹'+fmt(currentOrderPrice);
  document.getElementById('orderModalTitle').textContent = '● BUY ORDER';
  document.getElementById('orderModalTitle').style.color = 'var(--green)';
  document.getElementById('btnConfirmOrder').style.background = 'var(--green)';
  document.getElementById('btnConfirmOrder').dataset.type = 'BUY';
  document.getElementById('orderModal').classList.add('open');
}

function openSellModal(sym, price) {
  currentOrderSym   = sym || document.getElementById('sName').textContent;
  currentOrderPrice = price || LIVE[currentOrderSym] || 0;
  document.getElementById('orderSymbol').textContent  = currentOrderSym;
  document.getElementById('orderPrice').textContent   = '₹'+fmt(currentOrderPrice);
  document.getElementById('orderQty').value           = '1';
  document.getElementById('orderTotal').textContent   = '₹'+fmt(currentOrderPrice);
  document.getElementById('orderModalTitle').textContent = '● SELL ORDER';
  document.getElementById('orderModalTitle').style.color = 'var(--red)';
  document.getElementById('btnConfirmOrder').style.background = 'var(--red)';
  document.getElementById('btnConfirmOrder').dataset.type = 'SELL';
  document.getElementById('orderModal').classList.add('open');
}

document.addEventListener('DOMContentLoaded', () => {
  const qtyInput = document.getElementById('orderQty');
  if (qtyInput) {
    qtyInput.addEventListener('input', () => {
      const qty   = parseFloat(qtyInput.value) || 0;
      const total = (qty * currentOrderPrice).toFixed(2);
      document.getElementById('orderTotal').textContent = '₹'+fmt(total);
    });
  }
});

async function confirmOrder() {
  const qty  = parseFloat(document.getElementById('orderQty').value);
  const type = document.getElementById('btnConfirmOrder').dataset.type;
  if (!qty || qty < 1) { showToast('⚠ Enter valid quantity', 'alert'); return; }

  document.getElementById('btnConfirmOrder').textContent = 'PLACING...';

  try {
    const res = await fetch(`${BACKEND}/order`, {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify({ symbol:currentOrderSym, type, quantity:qty, price:currentOrderPrice })
    });
    const data = await res.json();
    closeModal('orderModal');
    if (data.success) {
      showToast(`✅ ${type} ORDER PLACED — ${currentOrderSym} × ${qty} @ ₹${fmt(currentOrderPrice)}`);
      // Auto add to portfolio if BUY
      if (type === 'BUY') {
        portfolio.push({ sym:currentOrderSym, qty, buyPrice:currentOrderPrice });
        savePortfolio();
        renderPortfolio();
      }
    } else {
      showToast(`⚠ Order failed: ${data.message}`, 'alert');
    }
  } catch {
    showToast('⚠ Backend offline — order not placed', 'alert');
  }
  document.getElementById('btnConfirmOrder').textContent = 'CONFIRM ORDER';
}

// ── SIGNAL ANALYSER ───────────────────────────────────────────
function loadStock(sym) {
  document.getElementById('stockInput').value = sym;
  analyzeStock();
}

async function analyzeStock() {
  const sym = document.getElementById('stockInput').value.trim().toUpperCase();
  if (!sym) return;
  document.getElementById('emptyState').style.display   = 'none';
  document.getElementById('signalBox').classList.remove('visible');
  document.getElementById('loadingOverlay').classList.add('visible');

  // Fetch live data from backend
  let liveData = null;
  if (backendOnline) {
    try {
      const res = await fetch(`${BACKEND}/live/${sym}`, { signal: AbortSignal.timeout(8000) });
      liveData  = await res.json();
      LIVE[sym] = liveData.price;
      if (!BASE_STOCKS[sym]) BASE_STOCKS[sym] = {};
      Object.assign(BASE_STOCKS[sym], {
        price:  liveData.price,
        change: liveData.change_pct,
        high:   liveData.high   || liveData.price*1.015,
        low:    liveData.low    || liveData.price*0.985,
        w52:    liveData.w52_high || liveData.price*1.18,
        vol:    fmtVol(liveData.volume||0),
        rsi:    liveData.rsi    || 50,
        macd:   liveData.macd   || 'NEUTRAL',
        signal: liveData.signal || 'HOLD',
        entry:  liveData.entry  || liveData.price*0.993,
        stop:   liveData.stop   || liveData.price*0.975,
        target: liveData.target || liveData.price*1.025,
      });
    } catch {}
  }

  setTimeout(() => {
    document.getElementById('loadingOverlay').classList.remove('visible');

    let d = BASE_STOCKS[sym];
    if (!d) {
      const price=Math.round(Math.random()*2000+300), chg=+((Math.random()-.45)*4).toFixed(1);
      const sigs=['BUY','SELL','HOLD'], macds=['BULLISH','BEARISH','NEUTRAL'], sig=sigs[Math.floor(Math.random()*3)];
      d={price,change:chg,high:+(price*1.015).toFixed(2),low:+(price*0.985).toFixed(2),w52:+(price*1.18).toFixed(2),vol:'1.8M',rsi:Math.round(Math.random()*40+30),macd:macds[Math.floor(Math.random()*3)],signal:sig,entry:+(price*(sig==='SELL'?1.006:0.993)).toFixed(2),stop:+(price*(sig==='SELL'?1.018:0.975)).toFixed(2),target:+(price*(sig==='SELL'?0.975:1.025)).toFixed(2)};
      LIVE[sym]=d.price; BASE_STOCKS[sym]=d;
    }

    const ltp = LIVE[sym]||d.price;
    document.getElementById('sName').textContent   = sym;
    document.getElementById('sPrice').textContent  = '₹'+fmt(ltp);
    const chgEl = document.getElementById('sChg');
    chgEl.textContent = (d.change>=0?'+':'')+d.change+'% today';
    chgEl.style.color = d.change>=0?'var(--green)':'var(--red)';
    document.getElementById('sHigh').textContent   = '₹'+fmt(d.high);
    document.getElementById('sLow').textContent    = '₹'+fmt(d.low);
    document.getElementById('s52w').textContent    = '₹'+fmt(d.w52);
    document.getElementById('sEntry').textContent  = '₹'+fmt(d.entry);
    document.getElementById('sStop').textContent   = '₹'+fmt(d.stop);
    document.getElementById('sTarget').textContent = '₹'+fmt(d.target);

    const badge = document.getElementById('sBadge');
    badge.className   = 'signal-badge '+d.signal.toLowerCase();
    badge.textContent = d.signal==='BUY'?'● BUY SIGNAL':d.signal==='SELL'?'● SELL SIGNAL':'● HOLD SIGNAL';

    document.getElementById('sRSI').textContent = d.rsi;
    const rb = document.getElementById('sRSIBar');
    rb.style.width = d.rsi+'%'; rb.style.background = d.rsi<40?'var(--red)':d.rsi>60?'var(--green)':'var(--amber)';

    document.getElementById('sMACD').textContent = d.macd;
    const mb = document.getElementById('sMACDBar');
    mb.style.background = d.macd==='BULLISH'?'var(--green)':d.macd==='BEARISH'?'var(--red)':'var(--amber)';
    mb.style.width = d.macd==='BULLISH'?'72%':d.macd==='BEARISH'?'35%':'55%';

    document.getElementById('sVol').textContent = d.vol;
    document.getElementById('sVolBar').style.width = (Math.random()*35+45)+'%';
    document.getElementById('sVolBar').style.background = 'var(--blue)';
    const mom = d.signal==='BUY'?'STRONG':d.signal==='SELL'?'WEAK':'NEUTRAL';
    document.getElementById('sMom').textContent = mom;
    const momW=d.signal==='BUY'?'75%':d.signal==='SELL'?'28%':'52%';
    const momC=d.signal==='BUY'?'var(--green)':d.signal==='SELL'?'var(--red)':'var(--amber)';
    document.getElementById('sMomBar').style.width = momW;
    document.getElementById('sMomBar').style.background = momC;

    const upside=(((d.target-d.entry)/d.entry)*100).toFixed(1);
    const rr=Math.abs((d.target-d.entry)/(d.entry-d.stop)).toFixed(1);
    const banner=document.getElementById('sBanner');
    if (d.signal==='BUY') {
      banner.className = 'best-banner';
      banner.innerHTML = `<b>Best Price to Buy: ₹${fmt(d.entry)}</b><br>Target: ₹${fmt(d.target)} &nbsp;|&nbsp; Stop Loss: ₹${fmt(d.stop)} &nbsp;|&nbsp; Upside: +${upside}%<br><em>Risk-Reward 1:${rr} · Based on RSI, MACD &amp; support levels</em>`;
    } else if (d.signal==='SELL') {
      banner.className = 'best-banner sell-banner';
      banner.innerHTML = `<b>Best Price to Short/Sell: ₹${fmt(d.entry)}</b><br>Target: ₹${fmt(d.target)} &nbsp;|&nbsp; Stop Loss: ₹${fmt(d.stop)} &nbsp;|&nbsp; Downside: ${upside}%<br><em>Bearish momentum · Resistance zone confirmed</em>`;
    } else {
      banner.className = 'best-banner';
      banner.style.borderLeftColor = 'var(--amber)';
      banner.innerHTML = `<b style="color:var(--amber)">Neutral Zone — Hold Position</b><br>Wait for breakout above ₹${fmt(d.high)} or below ₹${fmt(d.low)}<br><em>No clear momentum · Consolidation phase</em>`;
    }

    document.getElementById('alertSym').value = sym;

    // ── BUY / SELL BUTTONS ──────────────────────────────────
    const btnRow = document.getElementById('buySellRow');
    if (btnRow) {
      btnRow.innerHTML = `
        <button class="btn-buy-now" onclick="openBuyModal('${sym}', ${ltp})">
          ▲ BUY ${sym}
        </button>
        <button class="btn-sell-now" onclick="openSellModal('${sym}', ${ltp})">
          ▼ SELL ${sym}
        </button>`;
    }

    document.getElementById('signalBox').classList.add('visible');
    buildMainChart(sym);
    renderOrderBook(sym, ltp, liveData?.buy_depth, liveData?.sell_depth);
    updateLiveSignalPrice();
    // Fetch news for this specific stock
    fetchLiveNews(sym);
  }, 700);
}

function updateLiveSignalPrice() {
  const sym = document.getElementById('sName').textContent;
  if (!sym||sym==='--'||!LIVE[sym]) return;
  const el = document.getElementById('sPrice');
  if (el) el.textContent = '₹'+fmt(LIVE[sym]);
}

// ── PORTFOLIO ─────────────────────────────────────────────────
let portfolio = JSON.parse(localStorage.getItem('tp_portfolio')||'[]');
function savePortfolio() { localStorage.setItem('tp_portfolio',JSON.stringify(portfolio)); }

function openPortfolioModal() {
  document.getElementById('pfSym').value      = '';
  document.getElementById('pfQty').value      = '';
  document.getElementById('pfBuyPrice').value = '';
  document.getElementById('portfolioModal').classList.add('open');
}

function addPosition() {
  const sym   = document.getElementById('pfSym').value.trim().toUpperCase();
  const qty   = parseFloat(document.getElementById('pfQty').value);
  const price = parseFloat(document.getElementById('pfBuyPrice').value);
  if (!sym||!qty||!price) { showToast('⚠ Please fill all fields','alert'); return; }
  portfolio.push({sym,qty,buyPrice:price});
  savePortfolio(); closeModal('portfolioModal'); renderPortfolio();
  showToast(`✓ ${sym} × ${qty} added to portfolio`);
}

function removePosition(idx) {
  portfolio.splice(idx,1); savePortfolio(); renderPortfolio();
}

function renderPortfolio() {
  const body=document.getElementById('pfBody'), empty=document.getElementById('emptyPortfolio'), pfTbl=document.getElementById('pfTable');
  if (!portfolio.length) { pfTbl.style.display='none'; empty.style.display='block'; document.getElementById('pfSummary').innerHTML=''; return; }
  pfTbl.style.display=''; empty.style.display='none';
  let totalInvested=0, totalCurrent=0;
  body.innerHTML = portfolio.map((p,i)=>{
    const ltp=LIVE[p.sym]||p.buyPrice, invested=p.qty*p.buyPrice, current=p.qty*ltp;
    const pl=current-invested, ret=((pl/invested)*100).toFixed(2);
    totalInvested+=invested; totalCurrent+=current;
    const cls=pl>=0?'green':'red', sign=pl>=0?'+':'';
    return `<tr>
      <td style="color:var(--text);font-weight:500;letter-spacing:1px">${p.sym}</td>
      <td>${p.qty}</td><td>₹${fmt(p.buyPrice)}</td>
      <td class="${cls}" id="pf-ltp-${i}">₹${fmt(ltp)}</td>
      <td>₹${fmt(invested)}</td><td class="${cls}">₹${fmt(current)}</td>
      <td class="${cls}">${sign}₹${fmt(pl)}</td>
      <td class="${cls}">${sign}${ret}%</td>
      <td>
        <button class="btn-del" onclick="removePosition(${i})">✕</button>
        <button class="btn-sell-sm" onclick="openSellModal('${p.sym}',${ltp})" style="margin-left:4px">SELL</button>
      </td>
    </tr>`;
  }).join('');
  const totalPL=totalCurrent-totalInvested, totalRet=((totalPL/totalInvested)*100).toFixed(2);
  document.getElementById('pfSummary').innerHTML = `
    <div class="pf-stat"><div class="pf-stat-label">Total Invested</div><div class="pf-stat-val">₹${fmt(totalInvested)}</div></div>
    <div class="pf-stat"><div class="pf-stat-label">Current Value</div><div class="pf-stat-val">₹${fmt(totalCurrent)}</div></div>
    <div class="pf-stat"><div class="pf-stat-label">Total P&amp;L</div><div class="pf-stat-val" style="color:${totalPL>=0?'var(--green)':'var(--red)'}">${totalPL>=0?'+':''}₹${fmt(totalPL)}</div></div>
    <div class="pf-stat"><div class="pf-stat-label">Overall Return</div><div class="pf-stat-val" style="color:${totalPL>=0?'var(--green)':'var(--red)'}">${totalRet>=0?'+':''}${totalRet}%</div></div>`;
}

function updatePortfolioPrices() {
  portfolio.forEach((p,i)=>{
    const el=document.getElementById(`pf-ltp-${i}`);
    if (el&&LIVE[p.sym]) el.textContent='₹'+fmt(LIVE[p.sym]);
  });
  if (portfolio.length>0) renderPortfolio();
}

// ── PRICE ALERTS ──────────────────────────────────────────────
let alerts = JSON.parse(localStorage.getItem('tp_alerts')||'[]');
function saveAlerts() { localStorage.setItem('tp_alerts',JSON.stringify(alerts)); }

function openAlertModal() {
  const sym=document.getElementById('sName').textContent;
  if (sym&&sym!=='-') document.getElementById('alertSym').value=sym;
  document.getElementById('alertPrice').value='';
  document.getElementById('alertModal').classList.add('open');
}

function saveAlert() {
  const sym=document.getElementById('alertSym').value.trim().toUpperCase();
  const type=document.getElementById('alertType').value;
  const price=parseFloat(document.getElementById('alertPrice').value);
  if (!sym||!price) { showToast('⚠ Please fill all fields','alert'); return; }
  alerts.push({sym,type,price,triggered:false,id:Date.now()});
  saveAlerts(); closeModal('alertModal'); renderAlerts();
  showToast(`✓ Alert set: ${sym} ${type==='above'?'above':'below'} ₹${fmt(price)}`);
}

function removeAlert(id) {
  alerts=alerts.filter(a=>a.id!==id); saveAlerts(); renderAlerts();
}

function renderAlerts() {
  const el=document.getElementById('alertsList');
  if (!alerts.length) { el.innerHTML='<div class="empty-portfolio" style="padding:24px 18px">No alerts set. Click "+ NEW ALERT".</div>'; return; }
  el.innerHTML=alerts.map(a=>`
    <div class="alert-item">
      <div class="alert-info">
        <span class="alert-sym-badge">${a.sym}</span>
        <span class="alert-condition">${a.type==='above'?'▲ Above':'▼ Below'} ₹${fmt(a.price)}</span>
        ${a.triggered?'<span class="alert-triggered"> &nbsp;● TRIGGERED</span>':''}
      </div>
      <button class="btn-del" onclick="removeAlert(${a.id})">✕</button>
    </div>`).join('');
}

function checkAlerts() {
  alerts.forEach(a=>{
    if (a.triggered) return;
    const ltp=LIVE[a.sym]; if (!ltp) return;
    const hit=a.type==='above'?ltp>=a.price:ltp<=a.price;
    if (hit) {
      a.triggered=true; saveAlerts(); renderAlerts();
      showToast(`🔔 ALERT: ${a.sym} is ${a.type==='above'?'above':'below'} ₹${fmt(a.price)}! LTP: ₹${fmt(ltp)}`,'alert');
      if (Notification.permission==='granted')
        new Notification(`TradePulse Alert — ${a.sym}`,{body:`Price ${a.type==='above'?'crossed above':'fell below'} ₹${fmt(a.price)}. Current: ₹${fmt(ltp)}`});
    }
  });
}
if (Notification.permission==='default') Notification.requestPermission();

// ── LIVE NEWS FEED — GNews API + VADER Sentiment ──────────────
async function fetchLiveNews(stock) {
  const el = document.getElementById('newsFeed');
  if (!el) return;

  // Update news section header to show which stock
  const hdrEl = document.querySelector('.panel-header .news-hdr-title');
  if (hdrEl) {
    hdrEl.textContent = stock ? `${stock} NEWS` : 'MARKET NEWS';
    hdrEl.style.color = stock ? 'var(--green)' : 'var(--muted)';
  }

  // Show loading
  el.innerHTML = '<div style="padding:20px;text-align:center;color:var(--muted);font-family:var(--font-mono);font-size:12px;letter-spacing:1px">FETCHING ' + (stock ? stock + ' ' : '') + 'NEWS...</div>';

  try {
    const url = stock
      ? `${BACKEND}/news?stock=${stock}`
      : `${BACKEND}/news`;

    const res  = await fetch(url, { signal: AbortSignal.timeout(10000) });
    const data = await res.json();

    if (!data || !data.length) {
      el.innerHTML = '<div style="padding:20px;text-align:center;color:var(--muted);font-size:12px">No news found.</div>';
      return;
    }

    el.innerHTML = data.map(n => {
      const tagClass = n.tag === 'BULLISH' ? 'bull' : n.tag === 'BEARISH' ? 'bear' : 'neutral';
      const score    = n.score >= 0 ? '+' + n.score : n.score;
      return `
        <div class="news-card" onclick="window.open('${n.url}','_blank')" style="cursor:${n.url&&n.url!=='#'?'pointer':'default'}">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
            <div class="news-time">${n.time} · ${n.source}</div>
            <span style="font-family:var(--font-mono);font-size:10px;color:var(--muted)">${score}</span>
          </div>
          <div class="news-headline">${n.title}</div>
          ${n.desc ? `<div style="font-size:11px;color:var(--muted);margin-top:4px;line-height:1.5">${n.desc}</div>` : ''}
          <span class="news-tag ${tagClass}">${n.tag}</span>
        </div>`;
    }).join('');

  } catch (e) {
    // Fallback to static news if API fails
    renderStaticNews();
  }
}

function renderStaticNews() {
  const fallback = [
    {h:'Sensex surges 400 pts led by banking stocks; Nifty holds 22,600',tag:'bull',t:'09:35',src:'ET Markets'},
    {h:'RBI keeps repo rate unchanged at 6.5% — markets react positively',tag:'bull',t:'10:02',src:'Mint'},
    {h:'Reliance Industries Q3 results beat expectations — profit up 12%',tag:'bull',t:'10:18',src:'MoneyControl'},
    {h:'IT stocks under pressure; TCS, Infosys fall on weak guidance',tag:'bear',t:'10:35',src:'BS Markets'},
    {h:'FIIs turn net sellers in Indian equities, selling ₹1,200 Cr today',tag:'bear',t:'10:52',src:'CNBC TV18'},
    {h:'Tata Motors EV sales hit record high — stock up 2.3%',tag:'bull',t:'11:10',src:'Auto Desk'},
    {h:'HDFC Bank credit growth slows; analysts downgrade to HOLD',tag:'neutral',t:'11:28',src:'Mint'},
    {h:'Mid-cap index outperforms — Nifty Midcap up 1.4% today',tag:'bull',t:'12:00',src:'ET Now'},
    {h:'Bajaj Finance slides 1% after RBI digital lending scrutiny',tag:'bear',t:'12:22',src:'LiveMint'},
    {h:'Crude oil above $82/barrel — aviation stocks under pressure',tag:'bear',t:'13:00',src:'Reuters'},
  ];
  const el = document.getElementById('newsFeed');
  if (el) el.innerHTML = fallback.map(n=>`
    <div class="news-card">
      <div class="news-time">${n.t} IST · ${n.src}</div>
      <div class="news-headline">${n.h}</div>
      <span class="news-tag ${n.tag}">${n.tag==='bull'?'BULLISH':n.tag==='bear'?'BEARISH':'NEUTRAL'}</span>
    </div>`).join('');
}

function renderNews() { fetchLiveNews(); }

// ── MODALS & TOAST ────────────────────────────────────────────
function closeModal(id) { document.getElementById(id).classList.remove('open'); }
function showToast(msg, type='success') {
  const t=document.getElementById('toast');
  t.textContent=msg; t.className='toast show'+(type==='alert'?' alert-toast':type==='sell'?' sell-toast':'');
  clearTimeout(t._timer); t._timer=setTimeout(()=>t.classList.remove('show'),4000);
}

// ── MASTER TICK (2 seconds) ───────────────────────────────────
function masterTick() {
  if (!backendOnline) tickPrices();
  updateTicker(); liveUpdateChart();
  updateWatchlistPrices(); updateLiveSignalPrice(); updatePortfolioPrices();
  const sym=document.getElementById('sName').textContent;
  if (sym&&sym!=='-'&&LIVE[sym]) updateOrderBook(LIVE[sym]);
  checkAlerts();
}

document.getElementById('stockInput').addEventListener('keydown', e => {
  if (e.key==='Enter') analyzeStock();
});

// ── INITIALISE ────────────────────────────────────────────────
buildTicker();
renderIndices();
buildQuickPicks();
renderMovers();
renderWatchlist();
renderPortfolio();
renderAlerts();
renderNews();

// Fetch live prices FIRST then build UI
(async () => {
  await fetchAllPrices();
  buildTicker();
  renderWatchlist();
  fetchIndices();
  fetchGainers();
  fetchLosers();
  checkBackend();
})();

// Intervals
setInterval(masterTick,       2000);   // UI tick every 2s
setInterval(fetchAllPrices,  15000);   // Live prices every 15s
setInterval(fetchIndices,    15000);   // Live indices every 15s
setInterval(fetchGainers,    30000);   // Gainers/Losers every 30s
setInterval(fetchLosers,     30000);
setInterval(checkBackend,    10000);   // Backend health check every 10s
setInterval(fetchLiveNews,  300000);  // Refresh news every 5 minutes