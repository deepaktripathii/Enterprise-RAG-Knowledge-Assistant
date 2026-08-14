import type {
  ChatResponse,
  Conversation,
  ConversationDetail,
  DocumentRecord,
  EvaluationReport,
  Health,
  MonitoringSummary,
} from "./types";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    cache: "no-store",
    ...init,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail =
        typeof payload?.detail === "string"
          ? payload.detail
          : JSON.stringify(payload?.detail ?? payload);
    } catch {
      // keep the status text
    }
    throw new ApiError(response.status, detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),

  documents: () => request<DocumentRecord[]>("/documents"),

  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DocumentRecord>("/documents/upload", {
      method: "POST",
      body: form,
    });
  },

  processDocument: (id: string) =>
    request<{ document: DocumentRecord; message: string }>(
      `/documents/${id}/process`,
      { method: "POST" },
    ),

  deleteDocument: (id: string) =>
    request<void>(`/documents/${id}`, { method: "DELETE" }),

  chat: (payload: {
    question: string;
    conversation_id?: string | null;
    top_k?: number;
    document_ids?: string[];
  }) =>
    request<ChatResponse>("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  conversations: () => request<Conversation[]>("/conversations"),

  conversation: (id: string) =>
    request<ConversationDetail>(`/conversations/${id}`),

  deleteConversation: (id: string) =>
    request<void>(`/conversations/${id}`, { method: "DELETE" }),

  monitoring: () => request<MonitoringSummary>("/monitoring/summary"),

  evaluation: () => request<EvaluationReport>("/evaluation/latest"),

  runEvaluation: (topK = 5) =>
    request<EvaluationReport>(`/evaluation/run?top_k=${topK}`, {
      method: "POST",
    }),
};

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "Unexpected error";
}
