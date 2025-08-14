
const chatbox = document.getElementById("chatbox");
const chatform = document.getElementById("chatform");
const msg = document.getElementById("msg");
chatform.onsubmit = async (e)=>{
  e.preventDefault();
  let text = msg.value;
  if(!text) return;
  append("user", text);
  msg.value = "";
  let res = await fetch("/api/chat", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({message:text})});
  let j = await res.json();
  append("assistant", j.reply);
};
function append(role, text){
  let d = document.createElement("div");
  d.className = role;
  d.textContent = (role=== "user"? "You: " : "Assistant: ") + text;
  chatbox.appendChild(d);
  chatbox.scrollTop = chatbox.scrollHeight;
}
