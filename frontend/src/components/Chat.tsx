'use client'

import { useEffect, useState } from 'react'
import axios from 'axios'
import PulseLoader from 'react-spinners/PulseLoader'

type Recommendation = {
  assessment_id: string
  score?: number
  name?: string
  url?: string
  test_type_full?: string
  duration?: number
}

type ApiResponse = {
  recommended_assessments?: Recommendation[]
  final_results?: Recommendation[]
  clarification?: string
  trace_id?: string
  summary?: any
  [key: string]: any
}

const Chat: React.FC = () => {
  const [apiBase, setApiBase] = useState<string>('http://localhost:8000')
  const [query, setQuery] = useState<string>('')
  const [clarification, setClarification] = useState<string>('')
  const [history, setHistory] = useState<{ query: string; response: ApiResponse }[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [verbose, setVerbose] = useState<boolean>(false)
  const [useChat, setUseChat] = useState<boolean>(false) // toggle /chat for debug

  useEffect(() => {
    const saved = localStorage.getItem('api_base')
    if (saved) setApiBase(saved)
  }, [])

  const submitQuery = async () => {
    if (!query.trim()) {
      alert('Please enter a query')
      return
    }
    setLoading(true)
    localStorage.setItem('api_base', apiBase)
    try {
      const payload: any = { query }
      if (clarification.trim()) payload.clarification_answer = clarification.trim()
      if (verbose) payload.verbose = true
      const endpoint = useChat ? '/chat' : '/recommend'
      const res = await axios.post(`${apiBase.replace(/\/$/, '')}${endpoint}`, payload)
      setHistory((h) => [...h, { query, response: res.data }])
      setQuery('')
      setClarification('')
    } catch (err: any) {
      alert('Error: ' + err)
    }
    setLoading(false)
  }

  const latest = history[history.length - 1]
  const results =
    latest?.response?.recommended_assessments ||
    latest?.response?.final_results ||
    []

  return (
    <div>
      <div className="field">
        <label>API base</label>
        <input
          value={apiBase}
          onChange={(e) => setApiBase(e.target.value)}
          placeholder="http://localhost:8000"
        />
      </div>
      <div className="field">
        <label>Query</label>
        <textarea
          rows={4}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter query or JD"
        />
      </div>
      <div className="field">
        <label>Clarification (optional)</label>
        <input
          value={clarification}
          onChange={(e) => setClarification(e.target.value)}
          placeholder="If a clarification question was asked, answer here"
        />
      </div>
      <div className="controls">
        <label className="flex items-center gap-1 text-sm">
          <input type="checkbox" checked={verbose} onChange={(e) => setVerbose(e.target.checked)} />
          Verbose (debug)
        </label>
        <label className="flex items-center gap-1 text-sm">
          <input type="checkbox" checked={useChat} onChange={(e) => setUseChat(e.target.checked)} />
          Use /chat (includes summary/clarification)
        </label>
      </div>
      <button onClick={submitQuery} className="btn">
        {loading ? <PulseLoader size={8} color="#fff" /> : 'Submit'}
      </button>

      {latest && (
        <div className="mt-6">
          <p className="text-sm text-gray-600">Trace: {latest.response.trace_id || 'n/a'}</p>
          {latest.response.clarification && (
            <p className="text-sm text-yellow-700">Clarification requested: {latest.response.clarification}</p>
          )}
          <h3 className="text-lg font-semibold mt-2">Top results</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr className="bg-gray-100">
                  <th>Name</th>
                  <th>Duration</th>
                  <th>Type</th>
                  <th>Score</th>
                  <th>URL</th>
                </tr>
              </thead>
              <tbody>
                {results.map((ass: any, idx: number) => (
                  <tr key={idx}>
                    <td>{ass.name || ass.assessment_id}</td>
                    <td>{ass.duration ? `${ass.duration} min` : '—'}</td>
                    <td>{ass.test_type_full || ass.test_type || '—'}</td>
                    <td>{ass.score !== undefined ? ass.score.toFixed(3) : '—'}</td>
                    <td>
                      {ass.url ? (
                        <a href={ass.url} target="_blank" rel="noreferrer">
                          Link
                        </a>
                      ) : (
                        '—'
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {verbose && (
            <div className="mt-4">
              <h4 className="font-semibold">Raw response</h4>
              <pre className="bg-gray-900 text-green-400 p-2 rounded overflow-auto max-h-96">
                {JSON.stringify(latest.response, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default Chat
