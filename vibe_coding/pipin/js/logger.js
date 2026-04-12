// logger.js - sends errors to server /log endpoint

const LOG_ENDPOINT = '/log';

async function sendLog(level, message, extra = {}) {
  const entry = { level, message, ...extra, client_time: new Date().toISOString() };
  try {
    await fetch(LOG_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(entry),
    });
  } catch {
    // silently ignore — can't log the logger
  }
}

export function logError(message, extra = {}) { return sendLog('ERROR', message, extra); }
export function logWarn(message, extra = {})  { return sendLog('WARN',  message, extra); }
export function logInfo(message, extra = {})  { return sendLog('INFO',  message, extra); }
