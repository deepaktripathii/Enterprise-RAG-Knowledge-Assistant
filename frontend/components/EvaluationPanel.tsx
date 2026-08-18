"use client";

import { useEffect, useState } from "react";
import { api, errorMessage } from "@/lib/api";
import type { EvaluationReport } from "@/lib/types";

const METRICS: { key: keyof EvaluationReport["summary"]; label: string }[] = [
  { key: "retrieval_hit_rate", label: "Retrieval hit rate" },
  { key: "mrr", label: "MRR" },
  { key: "answer_keyword_coverage", label: "Answer keywords" },
  { key: "citation_accuracy", label: "Citation accuracy" },
  { key: "faithfulness", label: "Faithfulness" },
  { key: "answer_relevance", label: "Answer relevance" },
];

export default function EvaluationPanel({ onActivity }: { onActivity?: () => void }) {
  const [report, setReport] = useState<EvaluationReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .evaluation()
      .then((payload) => {
        if (!cancelled) setReport(payload);
      })
      .catch(() => {
        /* no report yet is expected on a fresh install */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      const payload = await api.runEvaluation(5);
      setReport(payload);
      onActivity?.();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card p-5">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-white">Retrieval evaluation</h2>
          <p className="text-sm text-slate-400">
            Fixed question set scored for retrieval, citation correctness and faithfulness.
          </p>
        </div>
        <button
          type="button"
          onClick={run}
          disabled={loading}
          className="rounded-lg border border-edge px-3 py-1.5 text-xs text-slate-200 transition hover:border-accent/60 hover:text-white disabled:opacity-50"
        >
          {loading ? "Running…" : "Run evaluation"}
        </button>
      </header>

      {error && (
        <p className="mt-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
          {error}
        </p>
      )}

      {!report && !error && (
        <p className="mt-4 rounded-lg border border-edge bg-white/[0.02] px-3 py-4 text-sm text-slate-500">
          No evaluation report yet. Run the evaluation after indexing documents.
        </p>
      )}

      {report && (
        <>
          <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            {METRICS.map(({ key, label }) => {
              const value = report.summary[key];
              return (
                <div key={key} className="rounded-xl border border-edge bg-white/[0.02] px-3 py-3">
                  <p className="text-[11px] uppercase tracking-wide text-slate-500">{label}</p>
                  <p className="mt-1 text-xl font-semibold text-white">
                    {typeof value === "number" ? value.toFixed(3) : "—"}
                  </p>
                </div>
              );
            })}
          </div>
          <p className="mt-3 text-xs text-slate-500">
            {report.summary.n_cases} questions · embedder {report.embedder} · top-k {report.top_k} ·
            generated {new Date(report.generated_at).toLocaleString()}
          </p>
        </>
      )}
    </section>
  );
}
