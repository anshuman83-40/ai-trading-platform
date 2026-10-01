/* ============================================================
   TRADEPULSE — LOGIN PAGE  |  login.js
   ============================================================ */

// Animate left panel stats
function animateStats() {
  const niftyEl  = document.getElementById('liveNifty');
  const sensexEl = document.getElementById('liveSensex');
  if (!niftyEl) return;
  setInterval(() => {
    const nifty  = (22600 + Math.random() * 40).toFixed(0);
    const sensex = (74500 + Math.random() * 80).toFixed(0);
    niftyEl.textContent  = Number(nifty).toLocaleString('en-IN');
    sensexEl.textContent = Number(sensex).toLocaleString('en-IN');
  }, 2000);
}

function switchTab(tab) {
  document.getElementById('loginForm').style.display  = tab === 'login'  ? 'block' : 'none';
  document.getElementById('signupForm').style.display = tab === 'signup' ? 'block' : 'none';
  document.getElementById('tabLogin').classList.toggle('active',  tab === 'login');
  document.getElementById('tabSignup').classList.toggle('active', tab === 'signup');
  document.getElementById('errMsg').textContent = '';
}

function showError(msg) {
  document.getElementById('errMsg').textContent = msg;
}

function handleLogin() {
  const email = document.getElementById('loginEmail').value.trim();
  const pass  = document.getElementById('loginPass').value;
  if (!email || !pass) { showError('⚠ Please fill all fields'); return; }

  const users = JSON.parse(localStorage.getItem('tp_users') || '[]');
  const user  = users.find(u => u.email === email && u.pass === pass);
  if (!user) { showError('⚠ Invalid email or password'); return; }

  localStorage.setItem('tp_loggedin', JSON.stringify({ name: user.name, email: user.email }));
  document.getElementById('loginBtnText').textContent = 'LOADING...';
  setTimeout(() => { window.location.href = 'index.html'; }, 600);
}

function handleSignup() {
  const name  = document.getElementById('signupName').value.trim();
  const email = document.getElementById('signupEmail').value.trim();
  const pass  = document.getElementById('signupPass').value;
  if (!name || !email || !pass) { showError('⚠ Please fill all fields'); return; }
  if (pass.length < 6) { showError('⚠ Password must be at least 6 characters'); return; }

  const users = JSON.parse(localStorage.getItem('tp_users') || '[]');
  if (users.find(u => u.email === email)) { showError('⚠ Email already registered'); return; }

  users.push({ name, email, pass });
  localStorage.setItem('tp_users', JSON.stringify(users));
  localStorage.setItem('tp_loggedin', JSON.stringify({ name, email }));
  setTimeout(() => { window.location.href = 'index.html'; }, 400);
}

function demoLogin() {
  localStorage.setItem('tp_loggedin', JSON.stringify({ name: 'Demo Trader', email: 'demo@tradepulse.in' }));
  window.location.href = 'index.html';
}

// Enter key support
document.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    const isLogin = document.getElementById('loginForm').style.display !== 'none';
    isLogin ? handleLogin() : handleSignup();
  }
});

animateStats();