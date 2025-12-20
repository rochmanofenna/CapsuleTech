import { useEffect, useRef, useState } from 'react'
import { useParams, Link } from 'react-router-dom'

const DEFAULT_API = 'https://capsuletech.rochmanofenna.workers.dev'
const API = window.API_BASE || import.meta.env.VITE_API_BASE || DEFAULT_API

function computeMaxSeq(list, current = 0) {
  let m = current
  for (const evt of list || []) {
    const seq = Number(evt.seq || evt.data?.seq || 0)
    if (seq > m) m = seq
  }
  return m
}

export default function RunDetailPage() {
  const { runId } = useParams()
  const [meta, setMeta] = useState(null)
  const [events, setEvents] = useState([])
  const [artifacts, setArtifacts] = useState([])
  const [status, setStatus] = useState('loading')
  const bottomRef = useRef(null)
  const lastSeqRef = useRef(0)
  const timerRef = useRef(null)

  useEffect(() => () => timerRef.current && clearInterval(timerRef.current), [])

  useEffect(() => {
    async function bootstrap() {
      setStatus('loading')
      try {
        const resp = await fetch(`${API}/api/runs/${encodeURIComponent(runId)}`)
        if (!resp.ok) throw new Error(`Run not found (${resp.status})`)
        const data = await resp.json()
        const runMeta = data.run || data
        setMeta(runMeta)
        setArtifacts(data.artifacts || [])
        const initialEvents = data.events || []
        setEvents(initialEvents)
        lastSeqRef.current = data.last_seq || computeMaxSeq(initialEvents)
        setStatus('live')
        if (timerRef.current) clearInterval(timerRef.current)
        timerRef.current = setInterval(poll, 1000)
      } catch (err) {
        setStatus(`error: ${err.message}`)
      }
    }

    async function poll() {
      try {
        const resp = await fetch(
          `${API}/api/runs/${encodeURIComponent(runId)}/events?after_seq=${lastSeqRef.current}`
        )
        if (!resp.ok) return
        const data = await resp.json()
        const newEvents = Array.isArray(data.events) ? data.events : Array.isArray(data) ? data : []
        if (!newEvents.length) return
        lastSeqRef.current = computeMaxSeq(newEvents, lastSeqRef.current)
        setEvents((prev) => [...prev, ...newEvents])
      } catch (err) {
        setStatus(`poll error: ${err.message}`)
      }
    }

    bootstrap()
    return () => timerRef.current && clearInterval(timerRef.current)
  }, [runId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events.length])

  return (
    <section className="details">
      <div className="section-header">
        <div>
          <h2>{runId}</h2>
          <Link to="/">Back to runs</Link>
        </div>
        <div>{status}</div>
      </div>
      {meta && (
        <div className="meta-grid">
          <div><strong>Backend:</strong> {meta.backend}</div>
          <div><strong>Policy:</strong> {meta.policy_id}</div>
          <div><strong>Track:</strong> {meta.track_id}</div>
          <div><strong>Created:</strong> {new Date(meta.created_at).toLocaleString()}</div>
        </div>
      )}
      {artifacts.length > 0 && (
        <div className="artifact-list">
          <h3>Artifacts</h3>
          <ul>
            {artifacts.map((artifact) => (
              <li key={artifact.name}>
                <a
                  href={`${API}/api/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifact.name)}`}
                  download
                >
                  {artifact.name} ({(artifact.size_bytes / 1024).toFixed(1)} KB)
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="events" id="event-log">
        {events.map((evt, idx) => (
          <div key={idx} className="event-card">
            <div><span className="type">{evt.type}</span> · seq {evt.seq}</div>
            <div><small>{evt.ts_ms ? new Date(evt.ts_ms).toLocaleString() : ''}</small></div>
            <pre>{JSON.stringify(evt.data, null, 2)}</pre>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </section>
  )
}
