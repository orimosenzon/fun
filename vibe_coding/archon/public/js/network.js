// Socket.IO client stub — used in hot-seat mode to verify connectivity.
// Real multiplayer sync arrives in the next milestone.

const Network = (() => {
  let socket = null;
  let connected = false;

  function init() {
    if (typeof io === 'undefined') {
      console.warn('socket.io not loaded — skipping connection');
      return;
    }
    socket = io();
    socket.on('connect', () => {
      connected = true;
      console.log(`[net] connected as ${socket.id}`);
      socket.emit('hello', { client: 'archon-web' });
    });
    socket.on('welcome', (data) => {
      console.log('[net] welcome:', data);
    });
    socket.on('disconnect', () => {
      connected = false;
      console.log('[net] disconnected');
    });
  }

  function isConnected() {
    return connected;
  }

  return { init, isConnected };
})();
