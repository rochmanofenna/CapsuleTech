import { Routes, Route, Link } from 'react-router-dom'
import RunsPage from './RunsPage.jsx'
import RunDetailPage from './RunDetailPage.jsx'
import './style.css'

export default function App() {
  return (
    <div className="app-shell">
      <header>
        <Link to="/" className="brand">
          <h1>CapsuleTech</h1>
        </Link>
        <p>Live capsule runs, manifests, and verifications.</p>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<RunsPage />} />
          <Route path="/runs/:runId" element={<RunDetailPage />} />
        </Routes>
      </main>
    </div>
  )
}
