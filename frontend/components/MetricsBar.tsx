"use client";

import type { Health, MonitoringSummary } from "@/lib/types";

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-xl border border-edge bg-white/[0.02] px-3 py-3">
      <p className="text-[11px] uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 text-xl font-semibold text-white">{value}</p>
      {hint && <p className="mt-0.5 text-[11px] text-slate-500">{hint}</p>}
    </div>
  );
}

export default function MetricsBar({
  summary,
  health,
}: {
  summary: MonitoringSummary | null;
  health: Health | null;
}) {
  const documents = summary?.documents ?? {};
  const completed = documents.completed ?? 0;
  const total = documents.total ?? 0;
  const vectors = summary?.vectors ?? health?.vectors ?? 0;
  const fallbackRate = summary?.fallback_rate;

  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
      <Stat label="Documents" value={`${completed}/${total}`} hint="indexed / uploaded" />
      <Stat label="Chunks" value={vectors.toLocaleString()} hint="vectors in Qdrant" />
      <Stat
        label="Questions"
        value={(summary?.total_queries ?? 0).toLocaleString()}
        hint={`${summary?.queries_last_24h ?? 0} in last 24h`}
      />
      <Stat
        label="Avg latency"
        value={summary?.avg_total_latency_ms ? `${Math.round(summary.avg_total_latency_ms)} ms` : "—"}
        hint={`p95 ${summary?.p95_total_latency_ms ?? 0} ms`}
      />
      <Stat
        label="Fallback rate"
        value={fallbackRate === null || fallbackRate === undefined ? "—" : `${(fallbackRate * 100).toFixed(0)}%`}
        hint="refused when context is missing"
      />
      <Stat
        label="Citations / answer"
        value={summary?.avg_citations !== null && summary?.avg_citations !== undefined ? summary.avg_citations.toFixed(1) : "—"}
        hint={health ? `${health.llm_provider}` : "provider"}
      />
    </div>
  );
}
