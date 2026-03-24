export type Assessment = {
  url: string;
  name: string;
  description?: string;
  duration?: number | null;
  remote_support?: "Yes" | "No" | string;
  adaptive_support?: "Yes" | "No" | string;
  test_type?: string[];
  score?: number | null;
};

export type DebugPayload = {
  plan?: any;
  fusion?: any;
  candidates?: any;
  rerank?: any;
  constraints?: any;
  [k: string]: any;
};

export type ChatResponse = {
  trace_id?: string;
  final_results?: Assessment[];
  recommended_assessments?: Assessment[];
  summary?: any;
  debug?: DebugPayload;
  clarification?: string;
};
