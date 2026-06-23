import type { VectorStoreCollection, IndexResult } from '../types'
import { API_BASE } from './apiService'

/**
 * Embed and index a saved chunk set into Qdrant.
 *
 * @param filename       Document filename (e.g. 'report.pdf').
 * @param chunksFilename Saved-chunk identifier — 'db:<id>' for DB backend or
 *                       the bare JSON filename for local backend.
 * @param collectionName Optional collection name override. When omitted, the
 *                       backend derives a deterministic name from the chunk config.
 */
export async function indexChunks(
  filename: string,
  chunksFilename: string,
  collectionName?: string | null,
): Promise<IndexResult> {
  const res = await fetch(`${API_BASE}/vector-store/index`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filename,
      chunks_filename: chunksFilename,
      collection_name: collectionName ?? null,
    }),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
    throw new Error(detail?.detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<IndexResult>
}

/** Return all Qdrant collections managed by Chunky. */
export async function listCollections(): Promise<VectorStoreCollection[]> {
  const res = await fetch(`${API_BASE}/vector-store/collections`)
  if (!res.ok) return []
  const data = await res.json().catch(() => ({ collections: [] }))
  return (data.collections ?? []) as VectorStoreCollection[]
}

/** Delete a named Qdrant collection. */
export async function deleteCollection(name: string): Promise<void> {
  const res = await fetch(
    `${API_BASE}/vector-store/collections/${encodeURIComponent(name)}`,
    { method: 'DELETE' },
  )
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
    throw new Error(detail?.detail ?? `HTTP ${res.status}`)
  }
}
