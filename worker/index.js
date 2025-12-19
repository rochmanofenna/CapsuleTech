const SAMPLE_RUNS = [
  {
    run_id: "demo-geom-001",
    backend: "geom",
    policy_id: "demo_policy_v1",
    track_id: "demo_geom_fast",
    created_at: "2025-12-19T05:55:00.000Z",
  },
  {
    run_id: "demo-risc0-001",
    backend: "risc0",
    policy_id: "demo_policy_v1",
    track_id: "demo_geom_fast",
    created_at: "2025-12-19T05:45:00.000Z",
  },
];

const SAMPLE_EVENTS = {
  "demo-geom-001": [
    {
      seq: 1,
      ts_ms: Date.now() - 1000 * 60 * 10,
      type: "run_started",
      data: { backend: "geom", track_id: "demo_geom_fast" },
    },
    {
      seq: 2,
      ts_ms: Date.now() - 1000 * 60 * 9,
      type: "proof_artifact",
      data: { size_bytes: 123456, path: "out/demo/adapter_proof.json" },
    },
    {
      seq: 3,
      ts_ms: Date.now() - 1000 * 60 * 8,
      type: "capsule_sealed",
      data: { capsule_hash: "cad1d397f70870f022e39fb8e274feb3" },
    },
  ],
  "demo-risc0-001": [
    {
      seq: 1,
      ts_ms: Date.now() - 1000 * 60 * 20,
      type: "run_started",
      data: { backend: "risc0", track_id: "demo_geom_fast" },
    },
  ],
};

function withCors(resp) {
  const headers = new Headers(resp.headers);
  headers.set("Access-Control-Allow-Origin", "*");
  headers.set("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  headers.set("Access-Control-Allow-Headers", "Content-Type,Authorization");
  return new Response(resp.body, { status: resp.status, headers });
}

function jsonResponse(data, status = 200) {
  return withCors(
    new Response(JSON.stringify(data), {
      status,
      headers: { "content-type": "application/json" },
    }),
  );
}

function notFound() {
  return withCors(new Response("Not found", { status: 404 }));
}

async function handleApi(request) {
  const url = new URL(request.url);
  if (url.pathname === "/api/health") {
    return jsonResponse({ ok: true, worker: "capsuletech" });
  }
  if (url.pathname === "/api/runs") {
    return jsonResponse({ runs: SAMPLE_RUNS });
  }
  const match = url.pathname.match(/^\/api\/runs\/(.+?)\/events$/);
  if (match) {
    const runId = decodeURIComponent(match[1]);
    const events = SAMPLE_EVENTS[runId] || [];
    return jsonResponse({ run_id: runId, events });
  }
  return notFound();
}

export default {
  async fetch(request) {
    if (request.method === "OPTIONS") {
      return withCors(new Response(null, { status: 204 }));
    }
    const url = new URL(request.url);
    if (url.pathname.startsWith("/api/")) {
      return handleApi(request);
    }
    return withCors(
      new Response(
        JSON.stringify({
          message: "CapsuleTech worker online",
          routes: ["/api/health", "/api/runs", "/api/runs/:id/events"],
        }),
        { headers: { "content-type": "application/json" } },
      ),
    );
  },
};
