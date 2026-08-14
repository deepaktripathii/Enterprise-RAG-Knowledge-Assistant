export type DocumentStatus = "pending" | "processing" | "completed" | "failed";

export interface DocumentRecord {
  id: string;
  filename: string;
  original_filename: string;
  content_type: string;
  size_bytes: number;
  sha256: string;
  page_count: number;
  chunk_count: number;
  status: DocumentStatus;
  error_message?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  processed_at?: string | null;
}

export interface Citation {
  index: number;
  document_id: string;
  filename: string;
  page: number;
  chunk_index: number;
  score: number;
  snippet: string;
}

export interface Source {
  document_id: string;
  filename: string;
  page: number;
  chunk_index: number;
  score: number;
}

export interface ChatResponse {
  conversation_id: string;
  message_id: number;
  question: string;
  answer: string;
  citations: Citation[];
  sources: Source[];
  grounded: boolean;
  fallback: boolean;
  warnings: string[];
  model: string;
  provider: string;
  retrieval_latency_ms: number;
  llm_latency_ms: number;
  total_latency_ms: number;
  usage: Record<string, number>;
  created_at: string;
}

export interface Message {
  id: number;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  model?: string | null;
  latency_ms?: number | null;
  created_at?: string | null;
}

export interface Conversation {
  id: string;
  title: string;
  message_count: number;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
  embedder: string;
  embedding_dimensions: number;
  llm_provider: string;
  llm_model: string;
  collection: string;
  vectors: number;
  documents: Record<string, number>;
  components: Record<string, string>;
  timestamp: string;
}

export interface MonitoringSummary {
  documents: Record<string, number>;
  vectors: number;
  conversations: number;
  total_queries: number;
  window_size: number;
  avg_retrieval_latency_ms: number | null;
  avg_llm_latency_ms: number | null;
  avg_total_latency_ms: number | null;
  p95_total_latency_ms: number | null;
  avg_top_score: number | null;
  fallback_rate: number | null;
  avg_citations: number | null;
  tokens: { prompt: number; completion: number; total: number };
  models: Record<string, number>;
  daily: { date: string; queries: number; avg_latency_ms: number }[];
  last_query_at: string | null;
  queries_last_24h: number;
}

export interface EvaluationSummary {
  retrieval_hit_rate: number;
  mrr: number;
  context_precision: number;
  answer_keyword_coverage: number;
  citation_accuracy: number;
  faithfulness: number;
  answer_relevance: number;
  grounded_rate: number;
  fallback_rate: number;
  avg_retrieval_latency_ms: number;
  avg_llm_latency_ms: number;
  avg_total_latency_ms: number;
  n_cases: number;
}

export interface EvaluationReport {
  generated_at: string;
  dataset: string;
  top_k: number;
  embedder: string;
  summary: EvaluationSummary;
  cases: {
    id: string;
    question: string;
    answer: string;
    citation_accuracy: number;
    faithfulness: number;
    hit_at_k: number;
  }[];
}
