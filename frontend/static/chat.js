# frontend/static/chat.js - Handles WebSocket connection
const socket = new WebSocket("ws://localhost:8000/chat");

document.getElementById("sendBtn").addEventListener("click", () => {
    let input = document.getElementById("chatInput").value;
    socket.send(input);
    document.getElementById("chatInput").value = "";
});

socket.onmessage = (event) => {
    let chatBox = document.getElementById("chatBox");
    chatBox.innerHTML += `<p>${event.data}</p>`;
};
