const mongoose = require('mongoose');

const Blog = require('../models/Blog.js');
const { runAgent, resumeAgent } = require('../services/agentRunner.js');
const MAX_BLOGS_PER_USER = 8;
const MAX_OPENAI_BLOGS_PER_USER = 2;

function setupSse(res) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  return (payload) => {
    res.write(`data: ${JSON.stringify(payload)}\n\n`);
  };
}

async function saveBlogIfPossible(req, result, topic) {
  if (!(mongoose.connection.readyState === 1 && req.user?.id)) {
    return null;
  }

  const existingBlogCount = await Blog.countDocuments({ user: req.user.id });
  if (existingBlogCount >= MAX_BLOGS_PER_USER) {
    const limitError = new Error(`You can only store up to ${MAX_BLOGS_PER_USER} blogs in history. Delete an older blog to save a new one.`);
    limitError.userMessage = limitError.message;
    throw limitError;
  }

  return Blog.create({
    user: req.user.id,
    topic,
    mode: result.mode || req.body.mode || 'hybrid',
    llmProvider: result.llmProvider || req.body.llmProvider || 'groq',
    llmModel: result.llmModel || req.body.llmModel || '',
    plan: result.plan,
    sections: result.sections,
    finalMarkdown: result.finalMarkdown,
    imageSpecs: result.imageSpecs || [],
    wordCount: result.wordCount || 0,
  });
}

async function enforceOpenAiGenerationLimit(req) {
  const selectedProvider = (req.body?.llmProvider || '').trim().toLowerCase();
  if (selectedProvider !== 'openai') {
    return;
  }

  if (!req.user?.id) {
    const authError = new Error('OpenAI generation requires an authenticated user session.');
    authError.userMessage = authError.message;
    throw authError;
  }

  if (mongoose.connection.readyState !== 1) {
    const unavailableError = new Error('OpenAI generation is temporarily unavailable. Please use Groq instead.');
    unavailableError.userMessage = unavailableError.message;
    throw unavailableError;
  }

  const openAiBlogCount = await Blog.countDocuments({
    user: req.user.id,
    llmProvider: 'openai',
  });

  if (openAiBlogCount >= MAX_OPENAI_BLOGS_PER_USER) {
    const limitError = new Error('You can generate at most 2 blogs with OpenAI on this account. Please use Groq for additional blogs.');
    limitError.userMessage = limitError.message;
    throw limitError;
  }
}

async function generateBlog(req, res) {
  const sendEvent = setupSse(res);

  try {
    const topic = req.body?.topic?.trim();
    if (!topic) {
      sendEvent({ type: 'error', message: 'Topic is required' });
      return;
    }

    await enforceOpenAiGenerationLimit(req);

    const result = await runAgent(req.body, (event) => {
      if (event?.type === 'step') {
        sendEvent(event);
      }
    });

    if (result.type === 'interrupt') {
      sendEvent({
        type: 'plan_review',
        sessionId: result.threadId,
        review: result.value,
      });
      return;
    }

    const savedBlog = await saveBlogIfPossible(req, result.blog, topic);

    sendEvent({
      type: 'complete',
      blog: savedBlog
        ? savedBlog.toObject()
        : {
            ...result.blog,
            topic,
            _id: null,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          },
    });
  } catch (error) {
    sendEvent({ type: 'error', message: error.userMessage || error.message || 'Generation failed' });
  } finally {
    res.end();
  }
}

async function reviewPlan(req, res) {
  const sendEvent = setupSse(res);

  try {
    const sessionId = req.body?.sessionId?.trim();
    const approved = Boolean(req.body?.approved);
    const topic = req.body?.topic?.trim();

    if (!sessionId) {
      sendEvent({ type: 'error', message: 'Session id is required to resume generation' });
      return;
    }

    const result = await resumeAgent(sessionId, approved, (event) => {
      if (event?.type === 'step') {
        sendEvent(event);
      }
    });

    if (result.type === 'interrupt') {
      sendEvent({
        type: 'plan_review',
        sessionId: result.threadId,
        review: result.value,
      });
      return;
    }

    const savedBlog = await saveBlogIfPossible(req, result.blog, topic || result.blog.topic || 'Untitled');

    sendEvent({
      type: 'complete',
      blog: savedBlog
        ? savedBlog.toObject()
        : {
            ...result.blog,
            _id: null,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
          },
    });
  } catch (error) {
    sendEvent({ type: 'error', message: error.userMessage || error.message || 'Generation failed' });
  } finally {
    res.end();
  }
}

module.exports = {
  generateBlog,
  reviewPlan,
};
