import { useState, useEffect, useRef, useCallback } from 'react'
import type { VectorStoreCollection, VectorSearchHit } from '../../types'
import { listCollections, searchChunks } from '../../services/vectorStoreApi'
import './VectorStoreModal.css'

interface Props {
  isOpen: boolean
  /** Pre-selected collection (set after "Embed & Index"). */
  defaultCollection?: string | null
  onClose: () => void
}

export default function VectorStoreModal({ isOpen, defaultCollection, onClose }: Props) {
  const [collections, setCollections] = useState<VectorStoreCollection[]>([])
  const [collection, setCollection] = useState<string>('')
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [hits, setHits] = useState<VectorSearchHit[]>([])
  const [searching, setSearching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loadingCols, setLoadingCols] = useState(false)
  const [expandedHit, setExpandedHit] = useState<number | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Load collections whenever the modal opens.
  useEffect(() => {
    if (!isOpen) return
    setLoadingCols(true)
    setError(null)
    listCollections()
      .then(cols => {
        setCollections(cols)
        // Auto-select: prefer the defaultCollection, else the first available.
        if (defaultCollection && cols.some(c => c.name === defaultCollection)) {
          setCollection(defaultCollection)
        } else if (cols.length > 0 && !collection) {
          setCollection(cols[0].name)
        }
      })
      .catch(err => setError(String(err)))
      .finally(() => setLoadingCols(false))
    // Focus the input after the modal renders.
    setTimeout(() => inputRef.current?.focus(), 50)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, defaultCollection])

  const handleSearch = useCallback(async () => {
    if (!query.trim() || !collection) return
    setSearching(true)
    setError(null)
    setHits([])
    setExpandedHit(null)
    try {
      const results = await searchChunks(collection, query.trim(), topK)
      setHits(results)
      if (results.length === 0) setError('No results found for this query.')
    } catch (err) {
      setError(`Search failed: ${err}`)
    } finally {
      setSearching(false)
    }
  }, [collection, query, topK])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) handleSearch()
    if (e.key === 'Escape') onClose()
  }

  const handleOverlay = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) onClose()
  }

  if (!isOpen) return null

  const scoreColor = (score: number) => {
    if (score >= 0.85) return 'var(--vs-score-high)'
    if (score >= 0.65) return 'var(--vs-score-mid)'
    return 'var(--vs-score-low)'
  }

  return (
    <div className="vs-overlay" onClick={handleOverlay}>
      <div className="vs-modal" role="dialog" aria-modal="true" aria-label="Semantic Search">
        {/* Header */}
        <div className="vs-header">
          <div className="vs-header-left">
            <span className="vs-icon">🔍</span>
            <h2 className="vs-title">Semantic Search</h2>
          </div>
          <button className="vs-close" onClick={onClose} title="Close">✕</button>
        </div>

        {/* Controls */}
        <div className="vs-controls">
          <div className="vs-row">
            <div className="vs-field vs-field--collection">
              <label className="vs-label">Collection</label>
              {loadingCols ? (
                <div className="vs-spinner-inline" />
              ) : (
                <select
                  className="vs-select"
                  value={collection}
                  onChange={e => { setCollection(e.target.value); setHits([]) }}
                  disabled={collections.length === 0}
                >
                  {collections.length === 0 && (
                    <option value="">— no collections —</option>
                  )}
                  {collections.map(c => (
                    <option key={c.name} value={c.name}>
                      {c.name} ({c.points_count.toLocaleString()} pts)
                    </option>
                  ))}
                </select>
              )}
            </div>
            <div className="vs-field vs-field--topk">
              <label className="vs-label">Top-K</label>
              <input
                type="number"
                className="vs-input-num"
                value={topK}
                min={1}
                max={50}
                onChange={e => setTopK(Math.min(50, Math.max(1, Number(e.target.value))))}
              />
            </div>
          </div>

          <div className="vs-search-row">
            <input
              ref={inputRef}
              className="vs-query-input"
              type="text"
              placeholder="Enter a natural-language query…"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={searching || !collection}
            />
            <button
              className="vs-search-btn"
              onClick={handleSearch}
              disabled={searching || !query.trim() || !collection}
            >
              {searching ? <span className="vs-spinner" /> : '↵ Search'}
            </button>
          </div>
        </div>

        {/* Error banner */}
        {error && (
          <div className="vs-error">
            <span>⚠️ {error}</span>
            <button className="vs-error-close" onClick={() => setError(null)}>✕</button>
          </div>
        )}

        {/* Results */}
        <div className="vs-results">
          {hits.length > 0 && (
            <div className="vs-results-header">
              <span className="vs-results-count">{hits.length} result{hits.length !== 1 ? 's' : ''}</span>
            </div>
          )}
          {hits.map((hit, i) => {
            const expanded = expandedHit === i
            const displayText = (hit.cleaned_chunk || hit.content).trim()
            const preview = displayText.slice(0, 180) + (displayText.length > 180 ? '…' : '')
            return (
              <div
                key={i}
                className={`vs-hit${expanded ? ' vs-hit--expanded' : ''}`}
                onClick={() => setExpandedHit(expanded ? null : i)}
              >
                <div className="vs-hit-header">
                  <span
                    className="vs-score"
                    style={{ color: scoreColor(hit.score) }}
                    title="Cosine similarity score"
                  >
                    {(hit.score * 100).toFixed(1)}%
                  </span>
                  {hit.title && (
                    <span className="vs-hit-title">{hit.title}</span>
                  )}
                  <span className="vs-hit-idx">#{hit.chunk_index + 1}</span>
                  <span className="vs-expand-icon">{expanded ? '▲' : '▼'}</span>
                </div>

                {!expanded && (
                  <p className="vs-hit-preview">{preview}</p>
                )}

                {expanded && (
                  <div className="vs-hit-body">
                    {hit.context && (
                      <div className="vs-hit-meta-row">
                        <span className="vs-meta-label">Context</span>
                        <span className="vs-meta-value">{hit.context}</span>
                      </div>
                    )}
                    <div className="vs-hit-content">{displayText}</div>
                    {hit.parent_content && (
                      <details className="vs-parent">
                        <summary>Parent chunk</summary>
                        <div className="vs-parent-content">{hit.parent_content}</div>
                      </details>
                    )}
                    {hit.keywords.length > 0 && (
                      <div className="vs-keywords">
                        {hit.keywords.map(kw => (
                          <span key={kw} className="vs-keyword">{kw}</span>
                        ))}
                      </div>
                    )}
                    <div className="vs-hit-footer">
                      <span className="vs-hit-file" title={hit.filename}>{hit.filename}</span>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
