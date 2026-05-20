const path = require('path');
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const PORT = process.env.PORT || 3000;

const app = express();
const server = http.createServer(app);
const io = new Server(server);

app.use(express.static(path.join(__dirname, '..', 'public')));

const rooms = new Map();

io.on('connection', (socket) => {
  console.log(`[+] client connected: ${socket.id}`);

  socket.on('hello', (payload) => {
    console.log(`[hello] from ${socket.id}:`, payload);
    socket.emit('welcome', { id: socket.id, message: 'connected to archon server' });
  });

  socket.on('disconnect', () => {
    console.log(`[-] client disconnected: ${socket.id}`);
  });
});

server.listen(PORT, () => {
  console.log(`archon server listening on http://localhost:${PORT}`);
});
