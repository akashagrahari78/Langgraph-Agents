import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { HiOutlineClipboardCopy, HiOutlineDownload, HiOutlineDocumentDownload, HiOutlinePencil } from 'react-icons/hi'
import axios from 'axios'
import ProgressTracker from '../components/ProgressTracker'
import MarkdownViewer from '../components/MarkdownViewer'
import { useGenerate } from '../hooks/useGenerate'
import useStore from '../store/useStore'
import { wordCount } from '../utils/helpers'

const EXAMPLE_TOPICS = [
  'State of Multimodal LLMs in 2026',
  'Building RAG pipelines with LangChain',
  'Kubernetes vs Docker Swarm in production',
  'WebAssembly: The future of web apps',
  'AI Code Assistants: A Deep Dive',
]

const MAX_TOPIC_WORDS = 30
const MAX_OPENAI_BLOGS = 2

const LLM_OPTIONS = [
  {
    provider: 'groq',
    label: 'Groq',
    model: 'llama-3.3-70b-versatile',
    description: 'Fast and strong for long technical drafts.',
  },
  {
    provider: 'openai',
    label: 'ChatGPT',
    model: 'gpt-4.1-mini',
    description: 'Balanced writing quality and reliability.',
  },
  {
    provider: 'claude',
    label: 'Claude',
    model: 'claude-3-5-sonnet-latest',
    description: 'Great for polished long-form writing.',
  },
]

export default function Generate() {
  const [topic, setTopic] = useState('')
  const [selectedLlm, setSelectedLlm] = useState(LLM_OPTIONS[0])
  const [audience, setAudience] = useState('developers')
  const [tone, setTone] = useState('professional')
  const [wordRange, setWordRange] = useState(2000)
  const [includeCode, setIncludeCode] = useState(true)
  const [includeCitations, setIncludeCitations] = useState(true)
  const [includeImages, setIncludeImages] = useState(false)
  const [placeholderIndex, setPlaceholderIndex] = useState(0)
  const [copied, setCopied] = useState(false)
  const [openAiUsageCount, setOpenAiUsageCount] = useState(0)

  const { generate, submitPlanReview, error } = useGenerate()
  const { token, isGenerating, progress, generatedBlog, pendingPlanReview, editMode, toggleEditMode, setEditMode } = useStore()
  const [editContent, setEditContent] = useState('')
  const textareaRef = useRef(null)
  const exportPreviewRef = useRef(null)

  useEffect(() => {
    const interval = setInterval(() => setPlaceholderIndex(i => (i + 1) % EXAMPLE_TOPICS.length), 3000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (generatedBlog?.finalMarkdown) setEditContent(generatedBlog.finalMarkdown)
  }, [generatedBlog])

  useEffect(() => {
    let cancelled = false

    const fetchOpenAiUsage = async () => {
      if (!token) {
        if (!cancelled) setOpenAiUsageCount(0)
        return
      }

      try {
        const res = await axios.get('/api/blogs', {
          headers: { Authorization: `Bearer ${token}` },
        })

        if (cancelled) return

        const openAiBlogs = Array.isArray(res.data)
          ? res.data.filter((blog) => (blog.llmProvider || '').toLowerCase() === 'openai').length
          : 0

        setOpenAiUsageCount(openAiBlogs)
      } catch (_) {
        if (!cancelled) setOpenAiUsageCount(0)
      }
    }

    fetchOpenAiUsage()

    return () => {
      cancelled = true
    }
  }, [token, generatedBlog])

  const topicWords = wordCount(topic)
  const isTopicLimitExceeded = topicWords > MAX_TOPIC_WORDS
  const isOpenAiLimitReached = openAiUsageCount >= MAX_OPENAI_BLOGS

  const handleTopicChange = (e) => {
    const nextValue = e.target.value
    const nextWordCount = wordCount(nextValue)

    if (nextWordCount > MAX_TOPIC_WORDS) {
      return
    }

    setTopic(nextValue)
  }

  const handleGenerate = () => {
    if (!topic.trim()) return
    setEditMode(false)
    generate({
      topic: topic.trim(),
      llmProvider: selectedLlm.provider,
      llmModel: selectedLlm.model,
      audience,
      tone,
      targetWordCount: wordRange,
      includeCode,
      includeCitations,
      includeImages,
    })
  }

  const handlePlanDecision = (approved) => {
    submitPlanReview({
      approved,
      topic: topic.trim(),
    })
  }

  const handleCopy = () => {
    const content = editMode ? editContent : generatedBlog?.finalMarkdown
    if (content) {
      navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleDownload = () => {
    const content = editMode ? editContent : generatedBlog?.finalMarkdown
    if (content) {
      const blob = new Blob([content], { type: 'text/markdown' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${generatedBlog?.plan?.blog_title || topic || 'blog'}.md`
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  const handleExportPdf = () => {
    const content = editMode ? editContent : generatedBlog?.finalMarkdown
    const renderedHtml = exportPreviewRef.current?.innerHTML
    if (!content || !renderedHtml) return

    const printWindow = window.open('', '_blank', 'width=900,height=700')
    if (!printWindow) return

    const safeTitle = (generatedBlog?.plan?.blog_title || topic || 'blog')
      .replace(/[<>:"/\\|?*]+/g, '')

    printWindow.document.write(`
      <!doctype html>
      <html>
        <head>
          <title>${safeTitle}</title>
          <style>
            * { box-sizing: border-box; }
            body { font-family: Arial, sans-serif; margin: 40px; color: #111827; line-height: 1.7; background: white; }
            .print-wrap { max-width: 900px; margin: 0 auto; }
            .print-title { font-size: 2rem; font-weight: 800; margin-bottom: 2rem; line-height: 1.2; }
            .markdown-body h1, .markdown-body h2, .markdown-body h3 { color: #111827; line-height: 1.25; margin-top: 1.75rem; margin-bottom: 0.85rem; }
            .markdown-body h1 { font-size: 2rem; }
            .markdown-body h2 { font-size: 1.45rem; }
            .markdown-body h3 { font-size: 1.15rem; }
            .markdown-body p, .markdown-body li { color: #374151; font-size: 0.98rem; margin-bottom: 0.9rem; }
            .markdown-body ul, .markdown-body ol { padding-left: 1.5rem; margin-bottom: 1rem; }
            .markdown-body blockquote { border-left: 4px solid #8b5cf6; background: #f5f3ff; color: #4b5563; padding: 0.9rem 1rem; border-radius: 0 10px 10px 0; margin: 1rem 0; }
            .markdown-body pre { white-space: pre-wrap; word-break: break-word; background: #111827; color: #f9fafb; padding: 16px; border-radius: 10px; overflow: hidden; margin-bottom: 1rem; }
            .markdown-body code:not(pre code) { background: #f3f4f6; color: #7c3aed; padding: 0.12rem 0.35rem; border-radius: 6px; font-size: 0.9em; }
            .markdown-body img { max-width: 100%; height: auto; margin: 1rem 0 0.5rem; }
            .markdown-body a { color: #2563eb; text-decoration: none; }
            .markdown-body hr { border: none; border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }
            @page { margin: 18mm 14mm; }
            @media print {
              body { margin: 0; }
              .print-wrap { max-width: none; }
            }
          </style>
        </head>
        <body>
          <div class="print-wrap">
            <div class="markdown-body">${renderedHtml}</div>
          </div>
        </body>
      </html>
    `)
    printWindow.document.close()
    printWindow.focus()
    printWindow.print()
  }

  const blogContent = editMode ? editContent : generatedBlog?.finalMarkdown
  const wc = wordCount(blogContent || '')
  const activeStep = progress.find((step) => step.status === 'active')
  const reviewPlan = pendingPlanReview?.plan

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="generate-container">
      <div className="generate-layout">
        {/* Left Panel — Input */}
        <div className="panel-left">
          <div className="generate-card">
            <div>
              <label className="input-label">Blog Topic</label>
              <textarea
                ref={textareaRef} value={topic} onChange={handleTopicChange}
                placeholder={EXAMPLE_TOPICS[placeholderIndex]} rows={3}
                className="input-textarea"
              />
              <div style={{ marginTop: '0.45rem', display: 'flex', justifyContent: 'space-between', gap: '0.75rem', alignItems: 'center' }}>
                <span style={{ fontSize: '0.72rem', color: 'var(--color-text-subtle)' }}>
                  {isTopicLimitExceeded ? 'Topic cannot exceed 30 words.' : 'Keep the topic short and focused.'}
                </span>
                <span style={{ fontSize: '0.72rem', color: topicWords >= MAX_TOPIC_WORDS ? '#f59e0b' : 'var(--color-text-subtle)', fontWeight: 600 }}>
                  {topicWords}/{MAX_TOPIC_WORDS} words
                </span>
              </div>
            </div>

            <div>
              <label className="input-label">LLM Provider</label>
              <div className="tone-group" style={{ flexWrap: 'wrap' }}>
                {LLM_OPTIONS.map((option) => {
                  const isDisabled = option.provider === 'openai' && isOpenAiLimitReached

                  return (
                  <button
                    key={option.provider}
                    type="button"
                    onClick={() => {
                      if (!isDisabled) setSelectedLlm(option)
                    }}
                    disabled={isDisabled}
                    className="tone-btn"
                    style={{
                      flex: '1 1 30%',
                      minWidth: '120px',
                      padding: '0.75rem',
                      borderRadius: '0.9rem',
                      backgroundColor: isDisabled
                        ? 'rgba(255,255,255,0.03)'
                        : selectedLlm.provider === option.provider
                          ? 'rgba(139, 92, 246, 0.12)'
                          : 'var(--color-bg-elevated)',
                      color: isDisabled
                        ? 'var(--color-text-subtle)'
                        : selectedLlm.provider === option.provider
                          ? '#e9d5ff'
                          : 'var(--color-text-primary)',
                      border: isDisabled
                        ? '1px solid rgba(255,255,255,0.05)'
                        : selectedLlm.provider === option.provider
                          ? '1px solid rgba(139, 92, 246, 0.4)'
                          : '1px solid var(--color-border)',
                      textAlign: 'left',
                      opacity: isDisabled ? 0.6 : 1,
                      cursor: isDisabled ? 'not-allowed' : 'pointer',
                    }}
                  >
                    <div style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '0.2rem' }}>{option.label}</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)', marginBottom: '0.3rem' }}>{option.model}</div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--color-text-subtle)', lineHeight: 1.4 }}>
                      {isDisabled ? 'OpenAI limit reached for this account. Use Groq instead.' : option.description}
                    </div>
                  </button>
                  )
                })}
              </div>
              {isOpenAiLimitReached && (
                <div style={{ marginTop: '0.55rem', fontSize: '0.72rem', color: '#f59e0b', fontWeight: 600 }}>
                  OpenAI is limited to 2 generated blogs per user. Please continue with Groq.
                </div>
              )}
            </div>

            <div>
              <div className="advanced-content">
                <div>
                  <label className="input-label" style={{ color: 'var(--color-text-muted)', fontSize: '0.75rem' }}>Audience</label>
                  <select value={audience} onChange={e => setAudience(e.target.value)} className="select-input">
                    <option value="developers">Developers</option>
                    <option value="beginners">Beginners</option>
                    <option value="technical-leads">Technical Leads</option>
                    <option value="general">General Audience</option>
                    <option value="executives">Executives / Business</option>
                  </select>
                </div>

                <div>
                  <label className="input-label" style={{ color: 'var(--color-text-muted)', fontSize: '0.75rem' }}>Tone</label>
                  <div className="tone-group">
                    {['professional', 'casual', 'academic'].map((t) => (
                      <button
                        key={t} onClick={() => setTone(t)}
                        className="tone-btn"
                        style={{
                          backgroundColor: tone === t ? 'rgba(139, 92, 246, 0.1)' : 'transparent',
                          color: tone === t ? '#c084fc' : 'var(--color-text-muted)',
                          border: tone === t ? '1px solid rgba(139, 92, 246, 0.4)' : '1px solid var(--color-border)',
                        }}
                      >
                        {t}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="word-count-header">
                    Target Word Count
                    <span style={{ color: '#d946ef', fontWeight: '600' }}>{wordRange.toLocaleString()}</span>
                  </label>
                  <input type="range" min={500} max={5000} step={100} value={wordRange} onChange={e => setWordRange(Number(e.target.value))} className="slider-input" />
                  <div className="slider-labels"><span>500</span><span>5,000</span></div>
                </div>

                <div className="toggle-group">
                  {[
                    { label: 'Include Code Snippets', value: includeCode, onChange: setIncludeCode },
                    { label: 'Include Citations', value: includeCitations, onChange: setIncludeCitations },
                    { label: 'Generate Images', value: includeImages, onChange: setIncludeImages },
                  ].map((toggle) => (
                    <label key={toggle.label} className="toggle-item">
                      <span className="toggle-label">{toggle.label}</span>
                      <div onClick={() => toggle.onChange(!toggle.value)} className={`toggle-switch ${toggle.value ? 'on' : ''}`} style={{ backgroundColor: !toggle.value ? 'var(--color-bg-hover)' : '' }}>
                        <div className="toggle-knob" />
                      </div>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            <button
              onClick={handleGenerate} disabled={isGenerating || !topic.trim() || Boolean(pendingPlanReview) || isTopicLimitExceeded}
              className={`btn-primary generate-btn ${isGenerating || !topic.trim() || pendingPlanReview || isTopicLimitExceeded ? 'btn-secondary' : ''}`}
            >
              {isGenerating ? 'Generating...' : 'Generate Blog'}
            </button>

            {error && <div className="error-badge">{error}</div>}
          </div>
        </div>

        {/* Right Panel — Output */}
        <div className="panel-right">
          {generatedBlog && (
            <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="right-toolbar">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                <span className="word-badge">{wc.toLocaleString()} words</span>
                <span className="word-badge" style={{ backgroundColor: 'rgba(6, 182, 212, 0.12)', color: '#67e8f9' }}>
                  {(generatedBlog?.llmProvider || selectedLlm.provider).toUpperCase()}
                </span>
              </div>
              <div className="toolbar-actions">
                <button onClick={handleCopy} className="action-btn" style={{ backgroundColor: 'var(--color-bg-elevated)', color: copied ? '#10b981' : 'var(--color-text-muted)' }}>
                  <HiOutlineClipboardCopy /> {copied ? 'Copied!' : 'Copy MD'}
                </button>
                <button onClick={handleDownload} className="action-btn" style={{ backgroundColor: 'var(--color-bg-elevated)', color: 'var(--color-text-muted)' }}>
                  <HiOutlineDownload /> Download
                </button>
                <button onClick={handleExportPdf} className="action-btn" style={{ backgroundColor: 'var(--color-bg-elevated)', color: 'var(--color-text-muted)' }}>
                  <HiOutlineDocumentDownload /> Export PDF
                </button>
                <button onClick={toggleEditMode} className="action-btn" style={{ backgroundColor: editMode ? 'rgba(139, 92, 246, 0.1)' : 'transparent', color: editMode ? '#c084fc' : 'var(--color-text-muted)' }}>
                  <HiOutlinePencil /> {editMode ? 'Preview' : 'Edit'}
                </button>
              </div>
            </motion.div>
          )}

          {progress.length > 0 && (isGenerating || pendingPlanReview) && (
            <div className="section-card">
              <h3 className="input-label" style={{ fontSize: '0.875rem' }}>Generation Progress</h3>
              {activeStep && (
                <div style={{ marginBottom: '1rem', padding: '0.9rem 1rem', borderRadius: '0.85rem', backgroundColor: 'rgba(139, 92, 246, 0.08)', border: '1px solid rgba(139, 92, 246, 0.18)' }}>
                  <div style={{ fontSize: '0.75rem', color: '#c084fc', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.35rem' }}>
                    Current Step
                  </div>
                  <div style={{ fontSize: '0.95rem', color: 'var(--color-text-primary)', fontWeight: 600 }}>
                    {activeStep.label}
                  </div>
                </div>
              )}
              <ProgressTracker steps={progress} />
            </div>
          )}

          {pendingPlanReview && reviewPlan && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="section-card" style={{ padding: '2rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', alignItems: 'flex-start', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                <div>
                  <h3 className="input-label" style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>Review Plan Before Drafting</h3>
                  <p style={{ margin: 0, color: 'var(--color-text-muted)', fontSize: '0.92rem', lineHeight: 1.6 }}>
                    LangGraph paused after the orchestrator step. Approve this outline to continue to section writing, or reject it to loop back and generate a fresh plan.
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                  <button onClick={() => handlePlanDecision(false)} disabled={isGenerating} className="action-btn" style={{ backgroundColor: 'var(--color-bg-elevated)', color: 'var(--color-text-primary)' }}>
                    No, regenerate plan
                  </button>
                  <button onClick={() => handlePlanDecision(true)} disabled={isGenerating} className="btn-primary" style={{ minWidth: '140px' }}>
                    Yes, continue
                  </button>
                </div>
              </div>

              <div style={{ display: 'grid', gap: '1rem' }}>
                <div style={{ padding: '1rem', borderRadius: '1rem', backgroundColor: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)' }}>
                  <div style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.4rem', color: 'var(--color-text-primary)' }}>
                    {reviewPlan.blog_title}
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <span className="word-badge">{reviewPlan.audience}</span>
                    <span className="word-badge">{reviewPlan.tone}</span>
                    <span className="word-badge">{reviewPlan.blog_kind?.replace('_', ' ')}</span>
                  </div>
                </div>

                {Array.isArray(reviewPlan.tasks) && reviewPlan.tasks.map((task) => (
                  <div key={task.id} style={{ padding: '1rem', borderRadius: '1rem', backgroundColor: 'var(--color-bg-elevated)', border: '1px solid var(--color-border)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', alignItems: 'center', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
                      <div style={{ fontWeight: 700, color: 'var(--color-text-primary)' }}>
                        {task.id}. {task.title}
                      </div>
                      <span className="word-badge">{task.target_words} words</span>
                    </div>
                    <p style={{ margin: '0 0 0.8rem 0', color: 'var(--color-text-muted)', lineHeight: 1.6 }}>
                      {task.goal}
                    </p>
                    <ul style={{ margin: 0, paddingLeft: '1.2rem', color: 'var(--color-text-muted)', lineHeight: 1.7 }}>
                      {Array.isArray(task.bullets) && task.bullets.map((bullet, index) => (
                        <li key={`${task.id}-${index}`}>{bullet}</li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {generatedBlog && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="section-card" style={{ padding: '2rem' }}>
              {editMode ? (
                <textarea value={editContent} onChange={e => setEditContent(e.target.value)} className="markdown-editor" />
              ) : (
                <MarkdownViewer content={blogContent} />
              )}
            </motion.div>
          )}

          {generatedBlog && (
            <div style={{ position: 'absolute', left: '-99999px', top: 0, width: '820px', pointerEvents: 'none', opacity: 0 }}>
              <div ref={exportPreviewRef}>
                <MarkdownViewer content={blogContent} animate={false} />
              </div>
            </div>
          )}

          {!isGenerating && !generatedBlog && !pendingPlanReview && (
            <div className="empty-state">
              <div style={{ textAlign: 'center' }}>
                <div className="empty-box"><HiOutlinePencil size={32} color="var(--color-text-subtle)" /></div>
                <p style={{ fontSize: '0.875rem', color: 'var(--color-text-muted)' }}>Enter a topic and click Generate to create your blog</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}
