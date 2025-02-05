const socket = new WebSocket("ws://localhost:8000/chat");

socket.onopen = () => {
    console.log("WebSocket connected!");
};

socket.onmessage = (event) => {
    console.log("Received from server:", event.data);
    let chatBox = document.getElementById("chatBox");
    chatBox.innerHTML += `<p>${event.data}</p>`;
};

socket.onerror = (error) => {
    console.error("WebSocket error:", error);
};

document.getElementById("sendBtn").addEventListener("click", () => {
    let input = document.getElementById("chatInput").value;
    console.log("Sending:", input);
    socket.send(input);
    document.getElementById("chatInput").value = "";
});

