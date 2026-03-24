"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Send,
  Settings,
  RefreshCw,
  Filter,
  Bug,
  Link as LinkIcon,
  Search,
  SlidersHorizontal,
  CheckCircle2,
  Loader2,
  XCircle,
} from "lucide-react";
import { fetchChat, fetchRecommend } from "@/lib/api";
import { useLocalStorage } from "@/lib/useLocalStorage";
import type { Assessment, ChatResponse, DebugPayload } from "@/types";

type HistoryItem = {
  id: string;
  query: string;
  response: ChatResponse | null;
  error?: string;
  ts: number;
};

const SAMPLE_PROMPTS = [
  "Java dev + collaboration + 40 minutes",
  "Sales graduate assessment for 60 minutes",
  "Culture fit assessment for COO, 60 minutes",
];

type Mode = "recommend" | "chat";
type TimelineStatus = "pending" | "in_progress" | "success" | "error";
type TimelineStep = { key: string; label: string; status: TimelineStatus };
const TIMELINE_ORDER: Array<{ key: string; label: string }> = [
  { key: "plan", label: "Plan" },
  { key: "retrieve", label: "Retrieve" },
  { key: "rerank", label: "Rerank" },
  { key: "constraints", label: "Finalize" },
];

const DEFAULT_API_BASE = "https://agamp-llm-recommendation-backend.hf.space";

function buildTimeline(backendTimeline: any, loading: boolean): TimelineStep[] {
  const status: Record<string, TimelineStatus> = {};
  TIMELINE_ORDER.forEach((s) => (status[s.key] = "pending"));

  if (Array.isArray(backendTimeline)) {
    for (const ev of backendTimeline) {
      const key = ev?.name;
      if (!key || !(key in status)) continue;
      const st = ev?.status;
      if (st === "start") status[key] = "in_progress";
      else if (st === "success") status[key] = "success";
      else if (st === "error") status[key] = "error";
    }
  } else if (loading) {
    // Request in-flight without timeline data yet
    status[TIMELINE_ORDER[0].key] = "in_progress";
  }

  return TIMELINE_ORDER.map((s) => ({
    ...s,
    status: status[s.key],
  }));
}

export default function Home() {
  const [apiBase, setApiBase] = useLocalStorage("api_base", DEFAULT_API_BASE);
  const [mode, setMode] = useLocalStorage<Mode>("mode", "recommend");
  const [verbose, setVerbose] = useLocalStorage("verbose", false);
  const [llmModel, setLlmModel] = useLocalStorage("llm_model", "Qwen/Qwen2.5-1.5B-Instruct");
  const [query, setQuery] = useState("");
  const [clarification, setClarification] = useState("");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [filters, setFilters] = useState({
    search: "",
    remote: "any" as "any" | "Yes" | "No",
    adaptive: "any" as "any" | "Yes" | "No",
    duration: "any" as "any" | "<=20" | "<=40" | "<=60" | "unknown",
    sort: "match" as "match" | "short" | "adaptive",
  });
  const [timelineSteps, setTimelineSteps] = useState<TimelineStep[]>(buildTimeline([], false));
  const defaultApi = "https://huggingface.co/spaces/AgamP/llm_recommendation_backend";

  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (history.length && activeIndex === null) {
      setActiveIndex(history.length - 1);
    }
  }, [history, activeIndex]);

  const activeItem = activeIndex !== null ? history[activeIndex] : null;
  const activeResults = (activeItem?.response?.recommended_assessments ||
    activeItem?.response?.final_results ||
    []) as Assessment[];
  const debug = activeItem?.response?.debug as DebugPayload | undefined;
  const backendTimeline = (debug as any)?.timeline;

  useEffect(() => {
    setTimelineSteps(buildTimeline(backendTimeline, loading));
  }, [backendTimeline, loading]);

  const filteredResults = useMemo(() => {
    let res = [...activeResults];
    const { search, remote, adaptive, duration, sort } = filters;
    if (search.trim()) {
      const q = search.toLowerCase();
      res = res.filter(
        (r) =>
          r.name?.toLowerCase().includes(q) ||
          r.description?.toLowerCase().includes(q) ||
          r.test_type?.some((t) => t.toLowerCase().includes(q))
      );
    }
    if (remote !== "any") {
      res = res.filter((r) => (r.remote_support || "").toLowerCase() === remote.toLowerCase());
    }
    if (adaptive !== "any") {
      res = res.filter((r) => (r.adaptive_support || "").toLowerCase() === adaptive.toLowerCase());
    }
    if (duration !== "any") {
      res = res.filter((r) => {
        const d = r.duration;
        if (d === null || d === undefined) return duration === "unknown";
        if (duration === "<=20") return d <= 20;
        if (duration === "<=40") return d <= 40;
        if (duration === "<=60") return d <= 60;
        return true;
      });
    }
    if (sort === "short") {
      res.sort((a, b) => (a.duration || 999) - (b.duration || 999));
    } else if (sort === "adaptive") {
      res.sort((a, b) => (b.adaptive_support === "Yes" ? 1 : 0) - (a.adaptive_support === "Yes" ? 1 : 0));
    }
    return res;
  }, [activeResults, filters]);

  const effectiveApiBase = apiBase?.trim() || DEFAULT_API_BASE;

  const send = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setTimelineSteps(buildTimeline([], true));
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const body: any = { query, verbose };
    if (clarification.trim()) body.clarification_answer = clarification.trim();
    if (mode === "recommend" && llmModel) body.llm_model = llmModel;
    const id = crypto.randomUUID();
    const ts = Date.now();
    setHistory((h) => [...h, { id, query, response: null, ts }]);
    try {
      const res =
        mode === "chat"
          ? await fetchChat(effectiveApiBase, body, controller.signal)
          : await fetchRecommend(effectiveApiBase, body, controller.signal);
      setHistory((h) =>
        h.map((item) => (item.id === id ? { ...item, response: res, error: undefined } : item))
      );
      setActiveIndex(history.length); // new item index
      setQuery("");
      setClarification("");
    } catch (err: any) {
      setHistory((h) => h.map((item) => (item.id === id ? { ...item, error: err.message } : item)));
    } finally {
      setLoading(false);
    }
  };

  const header = (
    <div className="flex items-center justify-between mb-3">
      <div>
        <h1 className="text-3xl font-semibold text-slate-900">SHL Assessment Recommender</h1>
        <p className="text-sm text-slate-600">Chat to get top-10 assessments. Filters and debug on the right.</p>
      </div>
      <div className="hidden md:flex items-center gap-2 text-xs text-slate-500">
        <RefreshCw size={16} /> Live against FastAPI backend
      </div>
    </div>
  );

  const settings = (
    <div className="flex flex-wrap gap-3 text-sm">
      <div className="flex items-center gap-2">
        <label className="font-medium">Mode</label>
        <select
          className="border rounded px-2 py-1"
          value={mode}
          onChange={(e) => setMode(e.target.value as Mode)}
        >
          <option value="recommend">/recommend</option>
          <option value="chat">/chat</option>
        </select>
      </div>
      <div className="flex items-center gap-2">
        <label className="font-medium">LLM</label>
        <input
          className="border rounded px-2 py-1"
          value={llmModel}
          onChange={(e) => setLlmModel(e.target.value)}
          placeholder="Qwen/Qwen2.5-1.5B-Instruct"
        />
      </div>
      <label className="flex items-center gap-2">
        <input type="checkbox" checked={verbose} onChange={(e) => setVerbose(e.target.checked)} />
        Verbose debug
      </label>
    </div>
  );

  const chatPanel = (
    <div className="flex flex-col h-full">
      <div className="flex flex-col gap-3 flex-1 overflow-hidden bg-white border rounded-xl shadow-sm p-4">
        <div className="flex items-center justify-between">
          <div className="text-lg font-semibold flex items-center gap-2">
            <Send size={18} /> Chat
          </div>
          <button
            onClick={() => {
              setQuery(SAMPLE_PROMPTS[0]);
            }}
            className="text-xs text-blue-600 hover:underline"
          >
            Use sample
          </button>
        </div>
        <div className="flex gap-2 items-center text-sm">
          <label className="font-medium min-w-[70px]">API base</label>
          <input
            className="border rounded px-2 py-1 w-full"
            value={apiBase || DEFAULT_API_BASE}
            onChange={(e) => setApiBase(e.target.value)}
          />
        </div>
        <textarea
          className="border rounded-lg p-3 w-full text-sm min-h-[140px] resize-none focus:ring-2 focus:ring-blue-200"
          placeholder="Enter job description or query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <div className="flex gap-2">
          {SAMPLE_PROMPTS.map((p) => (
            <button
              key={p}
              onClick={() => setQuery(p)}
              className="text-xs bg-slate-100 hover:bg-slate-200 px-2 py-1 rounded"
            >
              {p}
            </button>
          ))}
        </div>
        <div className="flex gap-3 items-center">
          <input
            className="border rounded px-2 py-1 text-sm flex-1"
            placeholder="Clarification (if asked)"
            value={clarification}
            onChange={(e) => setClarification(e.target.value)}
          />
          <button
            onClick={send}
            disabled={loading}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg flex items-center gap-2 hover:bg-blue-700 disabled:opacity-60"
          >
            <Send size={16} /> {loading ? "Sending..." : "Send"}
          </button>
          <button
            onClick={() => setVerbose(!verbose)}
            className="p-2 border rounded-lg hover:bg-slate-100"
            title="Toggle verbose debug"
          >
            <Bug size={16} />
          </button>
          <button
            onClick={() => setMode(mode === "recommend" ? "chat" : "recommend")}
            className="p-2 border rounded-lg hover:bg-slate-100"
            title="Toggle endpoint"
          >
            <Settings size={16} />
          </button>
        </div>
        {settings}
      </div>
      <div className="mt-3 bg-white border rounded-xl shadow-sm p-3 text-sm text-slate-600 max-h-48 overflow-auto">
        <div className="font-semibold mb-2">History</div>
        {history.length === 0 && <div className="text-slate-400">No queries yet.</div>}
        {history.map((h, idx) => (
          <button
            key={h.id}
            onClick={() => setActiveIndex(idx)}
            className={`block w-full text-left px-2 py-1 rounded ${
              idx === activeIndex ? "bg-blue-50 text-blue-700" : "hover:bg-slate-100"
            }`}
          >
            <div className="font-medium text-sm truncate">{h.query}</div>
            <div className="text-xs text-slate-500">{new Date(h.ts).toLocaleTimeString()}</div>
            {h.error && <div className="text-xs text-red-600">Error: {h.error}</div>}
          </button>
        ))}
      </div>
    </div>
  );

  const resultsPanel = (
    <div className="flex flex-col h-full">
      <div className="bg-white border rounded-xl shadow-sm p-4 flex flex-col gap-3 h-[75vh]">
        <div className="flex items-center justify-between">
          <div className="text-lg font-semibold flex items-center gap-2">
            <Filter size={18} /> Results
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <RefreshCw size={14} />
            Pipeline
          </div>
        </div>
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 overflow-x-auto pb-1">
            {timelineSteps.map((step, idx) => {
              const color =
                step.status === "success"
                  ? "bg-gradient-to-r from-emerald-50 to-emerald-100 text-emerald-800 border-emerald-200"
                  : step.status === "in_progress"
                  ? "bg-gradient-to-r from-blue-50 to-blue-100 text-blue-800 border-blue-200"
                  : step.status === "error"
                  ? "bg-gradient-to-r from-red-50 to-red-100 text-red-800 border-red-200"
                  : "bg-slate-50 text-slate-600 border-slate-200";
              const icon =
                step.status === "success" ? (
                  <CheckCircle2 size={14} />
                ) : step.status === "in_progress" ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : step.status === "error" ? (
                  <XCircle size={14} />
                ) : (
                  <div className="h-3 w-3 rounded-full bg-slate-300" />
                );
              return (
                <div key={step.key} className="flex items-center gap-2">
                  <div
                    className={`flex items-center gap-2 px-3 py-1.5 border rounded-full shadow-sm text-xs transition ${color}`}
                  >
                    {icon}
                    <span className="font-semibold whitespace-nowrap">{step.label}</span>
                  </div>
                  {idx < timelineSteps.length - 1 && <div className="h-[2px] w-6 bg-slate-200 rounded-full" />}
                </div>
              );
            })}
          </div>
          {!verbose && (
            <div className="text-[11px] text-slate-500">
              Turn on “Verbose debug” to see live pipeline status from the backend.
            </div>
          )}
        </div>
        <div className="flex items-center justify-between">
          <div className="relative">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-slate-400" />
            <input
              className="pl-8 pr-3 py-2 border rounded-lg text-sm"
              placeholder="Search results"
              value={filters.search}
              onChange={(e) => setFilters((f) => ({ ...f, search: e.target.value }))}
            />
          </div>
          <SlidersHorizontal size={16} className="text-slate-500" />
        </div>
        <div className="flex flex-wrap gap-3 text-xs">
          <select
            className="border rounded px-2 py-1"
            value={filters.remote}
            onChange={(e) => setFilters((f) => ({ ...f, remote: e.target.value as any }))}
          >
            <option value="any">Remote: Any</option>
            <option value="Yes">Remote: Yes</option>
            <option value="No">Remote: No</option>
          </select>
          <select
            className="border rounded px-2 py-1"
            value={filters.adaptive}
            onChange={(e) => setFilters((f) => ({ ...f, adaptive: e.target.value as any }))}
          >
            <option value="any">Adaptive: Any</option>
            <option value="Yes">Adaptive: Yes</option>
            <option value="No">Adaptive: No</option>
          </select>
          <select
            className="border rounded px-2 py-1"
            value={filters.duration}
            onChange={(e) => setFilters((f) => ({ ...f, duration: e.target.value as any }))}
          >
            <option value="any">Duration: Any</option>
            <option value="<=20">≤ 20 min</option>
            <option value="<=40">≤ 40 min</option>
            <option value="<=60">≤ 60 min</option>
            <option value="unknown">Unknown only</option>
          </select>
          <select
            className="border rounded px-2 py-1"
            value={filters.sort}
            onChange={(e) => setFilters((f) => ({ ...f, sort: e.target.value as any }))}
          >
            <option value="match">Sort: Best match</option>
            <option value="short">Sort: Shortest</option>
            <option value="adaptive">Sort: Adaptive first</option>
          </select>
        </div>
        <div className="flex-1 overflow-y-auto pr-1">
          <div className="grid md:grid-cols-2 lg:grid-cols-2 gap-3">
          {filteredResults.length === 0 && (
            <div className="text-sm text-slate-500">No results yet. Submit a query to see recommendations.</div>
          )}
          {filteredResults.map((r, idx) => (
            <div key={idx} className="border rounded-xl p-4 shadow-sm hover:shadow-md transition bg-slate-50">
              <div className="flex items-start justify-between gap-2">
                <a
                  href={r.url}
                  target="_blank"
                  rel="noreferrer"
                  className="font-semibold text-slate-900 hover:text-blue-600"
                >
                  {r.name || "Untitled"}
                </a>
                <button
                  className="text-slate-500 hover:text-blue-600"
                  onClick={() => r.url && navigator.clipboard.writeText(r.url)}
                >
                  <LinkIcon size={16} />
                </button>
              </div>
              <div className="flex flex-wrap gap-2 mt-2">
                {r.test_type?.map((t) => (
                  <span key={t} className="text-[11px] bg-blue-50 text-blue-700 px-2 py-1 rounded-full border border-blue-100">
                    {t}
                  </span>
                ))}
                <span className="text-[11px] bg-slate-100 text-slate-700 px-2 py-1 rounded-full border border-slate-200">
                  {r.duration ? `${r.duration} min` : "Duration unknown"}
                </span>
                <span className="text-[11px] bg-emerald-50 text-emerald-700 px-2 py-1 rounded-full border border-emerald-100">
                  Remote: {r.remote_support || "?"}
                </span>
                <span className="text-[11px] bg-indigo-50 text-indigo-700 px-2 py-1 rounded-full border border-indigo-100">
                  Adaptive: {r.adaptive_support || "?"}
                </span>
              </div>
              <p className="text-sm text-slate-700 mt-2 overflow-hidden text-ellipsis">{r.description || "No description."}</p>
            </div>
          ))}
          </div>
        </div>
      </div>
      {verbose && debug && (
        <div className="mt-3 bg-white border rounded-xl shadow-sm p-4">
          <div className="flex items-center gap-2 text-sm font-semibold mb-2">
            <Bug size={16} /> Debug
          </div>
          <div className="grid md:grid-cols-2 gap-3 text-xs">
            <div className="bg-slate-50 border rounded p-2">
              <div className="font-semibold mb-1">Plan</div>
              <pre className="overflow-auto max-h-48 text-slate-700">{JSON.stringify(debug.plan, null, 2)}</pre>
            </div>
            {debug.fusion && (
              <div className="bg-slate-50 border rounded p-2">
                <div className="font-semibold mb-1">Fusion</div>
                <pre className="overflow-auto max-h-48 text-slate-700">{JSON.stringify(debug.fusion, null, 2)}</pre>
              </div>
            )}
            {debug.candidates && (
              <div className="bg-slate-50 border rounded p-2 col-span-2">
                <div className="font-semibold mb-1">Top candidates</div>
                <pre className="overflow-auto max-h-60 text-slate-700">{JSON.stringify(debug.candidates, null, 2)}</pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <main className="min-h-screen bg-slate-100">
      <div className="app-shell py-6">
        {header}
        <div className="grid lg:grid-cols-2 gap-6 mt-4">
          {chatPanel}
          {resultsPanel}
        </div>
      </div>
    </main>
  );
}
