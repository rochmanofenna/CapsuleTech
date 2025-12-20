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

function normalize(payload, key) {
  if (payload && typeof payload === "object" && !Array.isArray(payload) && payload[key] !== undefined) {
    return payload;
  }
  return { [key]: payload };
}

async function proxyJson(path, env, wrapKey) {
  if (!env.RELAY_BASE) {
    return jsonResponse({
      error: "Relay base URL missing",
      hint: "Set RELAY_BASE in wrangler.toml or Worker environment",
    }, 500);
  }
  const target = new URL(path, env.RELAY_BASE);
  const upstream = await fetch(target.toString(), {
    headers: { accept: "application/json" },
  });
  const text = await upstream.text();
  let payload;
  try {
    payload = JSON.parse(text || "{}");
  } catch (err) {
    return jsonResponse({ error: "Relay returned non-JSON payload" }, 502);
  }
  if (wrapKey) {
    payload = normalize(payload, wrapKey);
  }
  return jsonResponse(payload, upstream.status);
}

async function proxyBinary(path, env) {
  if (!env.RELAY_BASE) {
    return jsonResponse({
      error: "Relay base URL missing",
      hint: "Set RELAY_BASE in wrangler.toml or Worker environment",
    }, 500);
  }
  const target = new URL(path, env.RELAY_BASE);
  const upstream = await fetch(target.toString());
  const headers = new Headers(upstream.headers);
  headers.set("Access-Control-Allow-Origin", "*");
  headers.set("Access-Control-Allow-Methods", "GET,OPTIONS");
  headers.set("Access-Control-Allow-Headers", "Content-Type,Authorization");
  return new Response(upstream.body, { status: upstream.status, headers });
}

async function handleApi(request, env) {
  const url = new URL(request.url);
  if (url.pathname === "/api/health") {
    return jsonResponse({ ok: true, worker: "capsuletech" });
  }
  if (url.pathname === "/api/runs") {
    return proxyJson(`/runs${url.search}`, env, "runs");
  }

  const artifactFileMatch = url.pathname.match(/^\/api\/runs\/(.+?)\/artifacts\/(.+)$/);
  if (artifactFileMatch) {
    const runId = decodeURIComponent(artifactFileMatch[1]);
    const filename = decodeURIComponent(artifactFileMatch[2]);
    return proxyBinary(`/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(filename)}`, env);
  }

  const artifactListMatch = url.pathname.match(/^\/api\/runs\/(.+?)\/artifacts$/);
  if (artifactListMatch) {
    const runId = decodeURIComponent(artifactListMatch[1]);
    return proxyJson(`/runs/${encodeURIComponent(runId)}/artifacts`, env, "artifacts");
  }

  const detailMatch = url.pathname.match(/^\/api\/runs\/(.+?)$/);
  if (detailMatch && !url.pathname.endsWith("/events")) {
    const runId = decodeURIComponent(detailMatch[1]);
    return proxyJson(`/runs/${encodeURIComponent(runId)}${url.search}`, env, "run");
  }

  const match = url.pathname.match(/^\/api\/runs\/(.+?)\/events$/);
  if (match) {
    const runId = decodeURIComponent(match[1]);
    const qs = url.search ? url.search : "";
    return proxyJson(`/runs/${encodeURIComponent(runId)}/events${qs}`, env, "events");
  }
  return notFound();
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return withCors(new Response(null, { status: 204 }));
    }
    const url = new URL(request.url);
    if (url.pathname.startsWith("/api/")) {
      return handleApi(request, env);
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
