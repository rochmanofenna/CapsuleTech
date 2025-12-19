export default {
  async fetch(request, env, ctx) {
    const body = JSON.stringify({
      schema: "capsule_worker_v1",
      message: "CapsuleBench worker is online",
      request: {
        method: request.method,
        url: request.url,
      },
    });
    return new Response(body, {
      status: 200,
      headers: {
        "content-type": "application/json; charset=utf-8",
        "cache-control": "no-store",
      },
    });
  },
};
