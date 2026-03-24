export type ChatRequest = {
  query: string;
  clarification_answer?: string | null;
  verbose?: boolean;
};

export type RecommendRequest = {
  query: string;
  llm_model?: string | null;
  verbose?: boolean;
};

export async function fetchChat(base: string, body: ChatRequest, signal?: AbortSignal) {
  return request(`${base.replace(/\/$/, "")}/chat`, body, signal);
}

export async function fetchRecommend(base: string, body: RecommendRequest, signal?: AbortSignal) {
  return request(`${base.replace(/\/$/, "")}/recommend`, body, signal);
}

async function request(url: string, body: any, signal?: AbortSignal) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: signal || controller.signal,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }
    return res.json();
  } finally {
    clearTimeout(timeout);
  }
}
