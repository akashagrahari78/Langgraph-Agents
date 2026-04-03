const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const activeSessions = new Map();

function buildFriendlyAgentError(rawMessage) {
  const message = rawMessage || 'Generation failed';

  if (message.includes('groq.RateLimitError') || message.includes('Rate limit reached for model')) {
    const retryMatch = message.match(/Please try again in\s+([^.]+(?:\.\d+)?s?)/i);
    const retryAfter = retryMatch ? retryMatch[1].trim() : null;
    const friendlyMessage = retryAfter
      ? `Groq rate limit reached for the blog model. Please try again in about ${retryAfter}.`
      : 'Groq rate limit reached for the blog model. Please try again later.';

    const error = new Error(friendlyMessage);
    error.userMessage = friendlyMessage;
    return error;
  }

  if (message.match(/Need more tokens\?/i)) {
    const friendlyMessage = 'The Groq daily token quota has been exhausted for this project. Please wait and try again later.';
    const error = new Error(friendlyMessage);
    error.userMessage = friendlyMessage;
    return error;
  }

  const error = new Error(message);
  error.userMessage = message;
  return error;
}

function getPythonCommand() {
  const repoRoot = path.resolve(__dirname, '..', '..', '..');
  const workspaceVenvPython = process.platform === 'win32'
    ? path.join(repoRoot, 'venv', 'Scripts', 'python.exe')
    : path.join(repoRoot, 'venv', 'bin', 'python');

  if (fs.existsSync(workspaceVenvPython)) {
    return workspaceVenvPython;
  }

  return process.platform === 'win32' ? 'py' : 'python3';
}

function settlePending(session, resolver) {
  if (!session.pending) return;
  const pending = session.pending;
  session.pending = null;
  resolver(pending);
}

function cleanupSession(sessionId) {
  const session = activeSessions.get(sessionId);
  if (!session) return;
  activeSessions.delete(sessionId);
}

function handleStdout(sessionId, chunk) {
  const session = activeSessions.get(sessionId);
  if (!session) return;

  session.stdoutBuffer += chunk;
  const lines = session.stdoutBuffer.split(/\r?\n/);
  session.stdoutBuffer = lines.pop() || '';

  for (const line of lines) {
    if (!line) continue;

    if (line.startsWith('STEP:')) {
      try {
        const step = JSON.parse(line.slice(5));
        session.onEvent?.({ type: 'step', stepIndex: step.index, status: step.status });
      } catch (_) {
        // ignore malformed step payloads
      }
      continue;
    }

    if (line.startsWith('INTERRUPT:')) {
      try {
        const interrupt = JSON.parse(line.slice(10));
        settlePending(session, ({ resolve }) => resolve({ type: 'interrupt', ...interrupt }));
      } catch (error) {
        settlePending(session, ({ reject }) => reject(buildFriendlyAgentError(error.message)));
      }
      continue;
    }

    if (line.startsWith('RESULT:')) {
      try {
        const result = JSON.parse(line.slice(7));
        settlePending(session, ({ resolve }) => resolve({ type: 'complete', blog: result }));
      } catch (error) {
        settlePending(session, ({ reject }) => reject(buildFriendlyAgentError(error.message)));
      }
      continue;
    }

    if (line.startsWith('ERROR:')) {
      try {
        const errorPayload = JSON.parse(line.slice(6));
        settlePending(session, ({ reject }) => reject(buildFriendlyAgentError(errorPayload.message)));
      } catch (error) {
        settlePending(session, ({ reject }) => reject(buildFriendlyAgentError(error.message)));
      }
    }
  }
}

function createSession(params, onEvent, pending = null) {
  const sessionId = params.sessionId || crypto.randomUUID();
  const agentScriptDir = path.resolve(__dirname, '..');
  const wrapperScript = path.resolve(__dirname, 'agent_wrapper.py');
  const payload = JSON.stringify({
    ...params,
    sessionId,
  });

  const child = spawn(getPythonCommand(), [wrapperScript, payload], {
    cwd: agentScriptDir,
    env: {
      ...process.env,
      PYTHONIOENCODING: 'utf-8',
    },
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  const session = {
    id: sessionId,
    child,
    stderr: '',
    stdoutBuffer: '',
    onEvent,
    pending,
  };

  activeSessions.set(sessionId, session);

  child.stdout.on('data', (data) => handleStdout(sessionId, data.toString()));
  child.stderr.on('data', (data) => {
    const current = activeSessions.get(sessionId);
    if (current) current.stderr += data.toString();
  });

  child.on('error', (err) => {
    const current = activeSessions.get(sessionId);
    if (!current) return;
    settlePending(current, ({ reject }) => reject(buildFriendlyAgentError(err.message || 'Failed to start agent process')));
    cleanupSession(sessionId);
  });

  child.on('close', (code) => {
    const current = activeSessions.get(sessionId);
    if (!current) return;

    if (code !== 0) {
      settlePending(current, ({ reject }) => reject(buildFriendlyAgentError(`Agent exited with code ${code}: ${current.stderr}`)));
    }

    cleanupSession(sessionId);
  });

  return session;
}

function waitForSession(session, onEvent) {
  session.onEvent = onEvent;
  return new Promise((resolve, reject) => {
    session.pending = { resolve, reject };
  });
}

async function runAgent(params, onEvent) {
  return new Promise((resolve, reject) => {
    createSession(params, onEvent, { resolve, reject });
  });
}

async function resumeAgent(sessionId, approved, onEvent) {
  const session = activeSessions.get(sessionId);
  if (!session) {
    throw buildFriendlyAgentError('Plan review session expired. Please generate again.');
  }

  const responsePromise = waitForSession(session, onEvent);
  session.child.stdin.write(`${JSON.stringify({ action: 'resume', approved })}\n`);
  return responsePromise;
}

function closeAgentSession(sessionId) {
  const session = activeSessions.get(sessionId);
  if (!session) return;
  session.child.stdin.write(`${JSON.stringify({ action: 'stop' })}\n`);
  cleanupSession(sessionId);
}

module.exports = {
  runAgent,
  resumeAgent,
  closeAgentSession,
};
