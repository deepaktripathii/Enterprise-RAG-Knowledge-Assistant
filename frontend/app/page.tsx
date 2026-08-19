"use client";

import { useCallback, useEffect, useState } from "react";
import ChatPanel from "@/components/ChatPanel";
import DocumentsPanel from "@/components/DocumentsPanel";
import EvaluationPanel from "@/components/EvaluationPanel";
import MetricsBar from "@/components/MetricsBar";
import { API_URL, api, errorMessage } from "@/lib/api";
import type { DocumentRecord, Health, MonitoringSummary } from "@/lib/types";

export default function HomePage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [summary, setSummary] = useState<MonitoringSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [healthPayload, documentList, summaryPayload] = await Promise.all([
        api.health(),
        api.documents(),
        api.monitoring(),
      ]);
      setHealth(healthPayload);
      setDocuments(documentList);
      setSummary(summaryPayload);
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [refresh]);

  const processing = documents.some(
    (document) => document.status === "pending" || document.status === "processing",
  );

  return (
    <main className="mx-auto max-w-7xl px-4 py-6 lg:px-8">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-accent/15 text-lg font-bold text-accent">
            SQ
          </div>
          <div>
            <h1 className="text-xl font-semibold text-white">SourceIQ</h1>
            <p className="text-xs text-slate-400">
              Enterprise AI knowledge intelligence · grounded answers with citations
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-[11px]">
          <span
            className={`pill ${
              health?.status === "ok"
                ? "bg-mint/10 text-mint"
                : health
                  ? "bg-amber-400/10 text-amber-300"
                  : "bg-rose-500/10 text-rose-300"
            }`}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-current" />
            {health?.status ?? "offline"}
          </span>
          {health && (
            <>
              <span className="pill bg-white/5 text-slate-300">{health.embedder}</span>
              <span className="pill bg-white/5 text-slate-300">{health.llm_model}</span>
              <span className="pill bg-white/5 text-slate-400">
                {health.embedding_dimensions}d
              </span>
            </>
          )}
        </div>
      </header>

      {error && (
        <div className="mt-4 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          <p className="font-medium">Cannot reach the API at {API_URL}</p>
          <p className="mt-1 text-xs text-rose-200/80">
            {error} — start it with{" "}
            <code className="rounded bg-black/30 px-1">uvicorn sourceiq.api.main:app --reload</code>.
          </p>
        </div>
      )}

      {processing && (
        <p className="mt-4 rounded-xl border border-accent/30 bg-accent/10 px-4 py-2 text-xs text-accent">
          Indexing in progress… this page refreshes automatically.
        </p>
      )}

      <div className="mt-5">
        <MetricsBar summary={summary} health={health} />
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-[minmax(320px,2fr)_minmax(0,3fr)]">
        <div className="min-h-[520px]">
          <DocumentsPanel documents={documents} onChanged={refresh} />
        </div>
        <div className="min-h-[520px]">
          <ChatPanel onActivity={refresh} />
        </div>
      </div>

      <div className="mt-5">
        <EvaluationPanel onActivity={refresh} />
      </div>

      <footer className="mt-6 flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-600">
        <span>
          SourceIQ v{health?.version ?? "1.0.0"} · Qdrant collection{" "}
          {health?.collection ?? "sourceiq_chunks"}
        </span>
        <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer" className="hover:text-slate-300">
          API documentation ↗
        </a>
      </footer>
    </main>
  );
}
