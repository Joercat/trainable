
document.getElementById("loginBtn").onclick = async ()=>{
  let user = document.getElementById("user").value;
  let pass = document.getElementById("pass").value;
  let res = await fetch("/api/admin/login", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({user, pass})});
  let j = await res.json();
  if(j.ok){ document.getElementById("login").style.display="none"; document.getElementById("dash").style.display="block"; }
  else alert("login failed");
};
document.getElementById("uploadBtn").onclick = async ()=>{
  let f = document.getElementById("fileinp").files[0];
  if(!f){ alert("pick"); return; }
  let fd = new FormData(); fd.append("file", f);
  let res = await fetch("/api/data/upload", {method:"POST", body: fd});
  let j = await res.json(); document.getElementById("status").textContent = JSON.stringify(j);
};
document.getElementById("trainBtn").onclick = async ()=>{
  let res = await fetch("/api/train/start", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({use_ratings:false, model:"ngram"})});
  let j = await res.json(); document.getElementById("status").textContent = JSON.stringify(j);
};
