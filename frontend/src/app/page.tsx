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
type Stats = {
  total: number;
  remote_yes: number;
  adaptive_yes: number;
  unique_job_levels: number;
  job_levels: string[];
  unique_languages: number;
  languages: string[];
  unique_test_types: number;
  test_types: string[];
};
const INITIAL_STATS: Stats = {
  total: 389,
  remote_yes: 389,
  adaptive_yes: 38,
  unique_job_levels: 10,
  job_levels: [
    "Entry-Level",
    "Graduate",
    "Mid-Professional",
    "Manager",
    "Director",
    "Executive",
    "Supervisor",
    "Front Line Manager",
    "Professional IC",
    "General Population",
  ],
  unique_languages: 12,
  languages: ["English (USA)", "English International", "French", "German", "Spanish", "Portuguese", "Italian", "Japanese", "Chinese", "Arabic", "Dutch", "Turkish"],
  unique_test_types: 8,
  test_types: ["Ability & Aptitude", "Biodata & Situational Judgement", "Competencies", "Development & 360", "Assessment Exercises", "Knowledge & Skills", "Personality & Behavior", "Simulations"],
};
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
  const [stats, setStats] = useState<Stats | null>(INITIAL_STATS);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [filters, setFilters] = useState({
    search: "",
    remote: "any" as "any" | "Yes" | "No",
    adaptive: "any" as "any" | "Yes" | "No",
    duration: "any" as "any" | "<=20" | "<=40" | "<=60" | "unknown",
    sort: "match" as "match" | "short" | "adaptive",
  });
  const [timelineSteps, setTimelineSteps] = useState<TimelineStep[]>(buildTimeline([], false));
  const abortRef = useRef<AbortController | null>(null);
  const timelineTimers = useRef<NodeJS.Timeout[]>([]);

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

  const clearSimTimeline = () => {
    timelineTimers.current.forEach((t) => clearTimeout(t));
    timelineTimers.current = [];
  };

  const startSimTimeline = () => {
    clearSimTimeline();
    setTimelineSteps(
      TIMELINE_ORDER.map((s, idx) => ({
        ...s,
        status: idx === 0 ? "in_progress" : "pending",
      }))
    );
    const seq: Array<[number, string]> = [
      [600, "plan"],
      [1200, "retrieve"],
      [1800, "rerank"],
      [2400, "constraints"],
    ];
    seq.forEach(([delay, key], i) => {
      const timer = setTimeout(() => {
        setTimelineSteps((prev) =>
          prev.map((step, idx) => {
            if (step.key === key) return { ...step, status: "success" };
            if (idx === i + 1) return { ...step, status: "in_progress" };
            return step;
          })
        );
      }, delay);
      timelineTimers.current.push(timer);
    });
  };

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
    startSimTimeline();
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
      clearSimTimeline();
      setLoading(false);
    }
  };

  const header = (
    <div className="flex items-center justify-between mb-2">
      <div>
        <h1 className="text-3xl font-semibold text-neutral-800">SHL Assessment Recommender</h1>
        <p className="text-sm text-neutral-600">Chat to get top-10 assessments. Filters and debug on the right.</p>
      </div>
    </div>
  );

  const statsPanel = (
    <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-2 mt-2">
      <div className="bg-gradient-to-r from-lime-50 to-white border border-lime-100 rounded-lg shadow-sm p-2">
        <div className="text-[11px] text-neutral-500">Assessments</div>
        <div className="text-xl font-semibold text-neutral-800">{stats?.total ?? "—"}</div>
      </div>
      <div className="bg-gradient-to-r from-lime-50 to-white border border-lime-100 rounded-lg shadow-sm p-2">
        <div className="text-[11px] text-neutral-500">Remote-supported</div>
        <div className="text-xl font-semibold text-neutral-800">{stats?.remote_yes ?? "—"}</div>
      </div>
      <div className="bg-gradient-to-r from-lime-50 to-white border border-lime-100 rounded-lg shadow-sm p-2">
        <div className="text-[11px] text-neutral-500">Adaptive</div>
        <div className="text-xl font-semibold text-neutral-800">{stats?.adaptive_yes ?? "—"}</div>
      </div>
      <div className="bg-gradient-to-r from-lime-50 to-white border border-lime-100 rounded-lg shadow-sm p-2 col-span-1 lg:col-span-2">
        <div className="text-[11px] text-neutral-500">
          Job levels ({stats?.unique_job_levels ?? "—"})
        </div>
        <div className="text-xs text-neutral-700 truncate">
          {stats?.job_levels && stats.job_levels.length ? stats.job_levels.join(", ") : "N/A"}
        </div>
      </div>
      <div className="bg-gradient-to-r from-lime-50 to-white border border-lime-100 rounded-lg shadow-sm p-2">
        <div className="text-[11px] text-neutral-500">Test types</div>
        <div className="text-xs text-neutral-700 truncate">
          {stats?.test_types && stats.test_types.length ? stats.test_types.join(", ") : "N/A"}
        </div>
      </div>
    </div>
  );

  const settings = (
    <div className="flex flex-wrap gap-3 text-sm">
      <div className="flex items-center gap-2">
        <label className="font-medium">API</label>
        <select
          className="border rounded px-2 py-1"
          value={mode}
          onChange={(e) => setMode(e.target.value as Mode)}
        >
          <option value="recommend">/recommend</option>
          <option value="chat">/chat</option>
          <option value="health">/health</option>
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
      <div className="flex flex-col gap-3 flex-1 overflow-hidden bg-white border border-neutral-300 rounded-xl shadow-sm p-4">
        <div className="flex items-center justify-between">
          <div className="text-lg font-semibold flex items-center gap-2">
            <Send size={18} /> Chat
          </div>
          <button
            onClick={() => {
              setQuery(SAMPLE_PROMPTS[0]);
            }}
            className="text-xs text-neutral-700 hover:underline"
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
        {loading && (
          <div className="text-xs text-neutral-600 bg-neutral-100 border border-neutral-200 rounded px-3 py-2">
            Processing your request. This may take a few seconds—thanks for your patience.
          </div>
        )}
        <textarea
          className="border rounded-lg p-3 w-full text-sm min-h-[140px] resize-none focus:ring-2 focus:ring-lime-200"
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
              className="text-xs bg-neutral-100 text-neutral-800 hover:bg-neutral-200 px-2 py-1 rounded border border-neutral-200"
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
            className="bg-neutral-800 text-white px-4 py-2 rounded-lg flex items-center gap-2 hover:bg-neutral-900 disabled:opacity-60 shadow-sm"
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
          <button
            onClick={() => {
              setQuery(SAMPLE_PROMPTS[0]);
              setClarification("");
            }}
            className="p-2 border rounded-lg hover:bg-slate-100"
            title="Reset"
          >
            <RefreshCw size={16} />
          </button>
        </div>
        {settings}
      </div>
      <div className="mt-2 bg-white border border-neutral-200 rounded-xl shadow-sm p-3 text-sm text-neutral-600 max-h-36 overflow-auto">
        <div className="font-semibold mb-2">History</div>
        {history.length === 0 && <div className="text-slate-400">No queries yet.</div>}
        {history.map((h, idx) => (
          <button
            key={h.id}
            onClick={() => setActiveIndex(idx)}
            className={`block w-full text-left px-2 py-1 rounded ${
              idx === activeIndex ? "bg-neutral-100 text-neutral-900 border border-neutral-200" : "hover:bg-slate-100"
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
      <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-3 flex flex-col gap-2 h-[68vh]">
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
                  ? "bg-gradient-to-r from-lime-50 to-lime-100 text-lime-800 border-lime-200"
                  : step.status === "in_progress"
                  ? "bg-gradient-to-r from-neutral-100 to-neutral-200 text-neutral-800 border-neutral-200"
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
            <div key={idx} className="border rounded-xl p-4 shadow-sm hover:shadow-md transition bg-slate-50 border-slate-200">
              <div className="flex items-start justify-between gap-2">
                <a
                  href={r.url}
                  target="_blank"
                  rel="noreferrer"
                  className="font-semibold text-neutral-900 hover:text-neutral-700"
                >
                  {r.name || "Untitled"}
                </a>
                <button
                  className="text-slate-500 hover:text-neutral-700"
                  onClick={() => r.url && navigator.clipboard.writeText(r.url)}
                >
                  <LinkIcon size={16} />
                </button>
              </div>
              <div className="flex flex-wrap gap-2 mt-2">
                {r.test_type?.map((t) => (
                  <span key={t} className="text-[11px] bg-neutral-100 text-neutral-800 px-2 py-1 rounded-full border border-neutral-200">
                    {t}
                  </span>
                ))}
                <span className="text-[11px] bg-slate-100 text-slate-700 px-2 py-1 rounded-full border border-slate-200">
                  {r.duration ? `${r.duration} min` : "Duration unknown"}
                </span>
                <span className="text-[11px] bg-lime-50 text-lime-700 px-2 py-1 rounded-full border border-lime-100">
                  Remote: {r.remote_support || "?"}
                </span>
                <span className="text-[11px] bg-neutral-100 text-neutral-800 px-2 py-1 rounded-full border border-neutral-200">
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
    <main className="min-h-screen bg-gradient-to-b from-white via-slate-50 to-slate-100">
      <div className="app-shell py-4">
        {header}
        {statsPanel}
        <div className="grid lg:grid-cols-2 gap-4 mt-3">
          {chatPanel}
          {resultsPanel}
        </div>
      </div>
    </main>
  );
}
