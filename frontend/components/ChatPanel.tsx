"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { api, errorMessage } from "@/lib/api";
import type { ChatResponse } from "@/lib/types";

type Turn =
  | { question: string; state: "loading" }
  | { question: string; state: "error"; error: string }
  | { question: string; state: "done"; answer: ChatResponse };

const SUGGESTIONS = [
  "How many days per week can I work remotely?",
  "What is the minimum password length?",
  "What is the hotel limit in metro cities?",
  "What is the P1 response time?",
];

export default function ChatPanel({ onActivity }: { onActivity: () => void }) {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  async function ask(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setQuestion("");
    setTurns((previous) => [...previous, { question: trimmed, state: "loading" }]);

    try {
      const answer = await api.chat({
        question: trimmed,
        conversation_id: conversationId,
      });
      setConversationId(answer.conversation_id);
      setTurns((previous) => {
        const next = [...previous];
        next[next.length - 1] = { question: trimmed, state: "done", answer };
        return next;
      });
      onActivity();
    } catch (err) {
      setTurns((previous) => {
        const next = [...previous];
        next[next.length - 1] = {
          question: trimmed,
          state: "error",
          error: errorMessage(err),
        };
        return next;
      });
    } finally {
      setBusy(false);
    }
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    ask(question);
  }

  function reset() {
    setTurns([]);
    setConversationId(null);
  }

  return (
    <section className="card flex h-full flex-col p-5">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white">Ask your documents</h2>
          <p className="text-sm text-slate-400">
            Answers are grounded in retrieved passages with page-level citations.
          </p>
        </div>
        <button
          type="button"
          onClick={reset}
          className="rounded-lg border border-edge px-2.5 py-1 text-xs text-slate-300 transition hover:border-accent/60 hover:text-white"
        >
          New chat
        </button>
      </header>

      <div className="mt-4 flex-1 space-y-4 overflow-y-auto pr-1">
        {turns.length === 0 && (
          <div className="rounded-xl border border-edge bg-white/[0.02] p-4">
            <p className="text-sm text-slate-300">Try one of these:</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => ask(suggestion)}
                  className="rounded-full border border-edge bg-white/[0.03] px-3 py-1.5 text-xs text-slate-300 transition hover:border-accent/60 hover:text-white"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((turn, index) => (
          <div key={`${turn.question}-${index}`} className="space-y-2">
            <div className="ml-auto max-w-[85%] rounded-2xl rounded-br-sm bg-accent/15 px-3 py-2 text-sm text-slate-100">
              {turn.question}
            </div>

            {turn.state === "loading" && (
              <div className="max-w-[90%] rounded-2xl rounded-bl-sm bg-white/[0.04] px-3 py-2 text-sm text-slate-400">
                Retrieving passages and drafting a grounded answer…
              </div>
            )}

            {turn.state === "error" && (
              <div className="max-w-[90%] rounded-2xl rounded-bl-sm border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
                {turn.error}
              </div>
            )}

            {turn.state === "done" && (
              <div className="max-w-[95%] space-y-2">
                <div
                  className={`rounded-2xl rounded-bl-sm border px-3 py-2 text-sm leading-relaxed ${
                    turn.answer.fallback
                      ? "border-amber-400/30 bg-amber-400/10 text-amber-100"
                      : "border-edge bg-white/[0.04] text-slate-100"
                  }`}
                >
                  {turn.answer.answer}
                </div>

                {turn.answer.citations.length > 0 && (
                  <div className="space-y-1.5">
                    <p className="text-[11px] uppercase tracking-wide text-slate-500">Sources</p>
                    {turn.answer.citations.map((citation) => (
                      <details
                        key={`${citation.document_id}-${citation.page}-${citation.index}`}
                        className="rounded-lg border border-edge bg-white/[0.02] px-3 py-2"
                      >
                        <summary className="cursor-pointer text-xs text-slate-300">
                          <span className="font-medium text-accent">[{citation.index}]</span>{" "}
                          {citation.filename} · page {citation.page}
                          <span className="ml-2 text-slate-500">
                            score {citation.score.toFixed(2)}
                          </span>
                        </summary>
                        <p className="mt-2 text-xs leading-relaxed text-slate-400">
                          {citation.snippet}
                        </p>
                      </details>
                    ))}
                  </div>
                )}

                <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
                  <span
                    className={`pill ${
                      turn.answer.grounded
                        ? "bg-mint/10 text-mint"
                        : "bg-amber-400/10 text-amber-300"
                    }`}
                  >
                    {turn.answer.grounded ? "grounded" : "not grounded"}
                  </span>
                  <span>{turn.answer.model}</span>
                  <span>· {Math.round(turn.answer.total_latency_ms)} ms</span>
                  {turn.answer.usage.total_tokens > 0 && (
                    <span>· {turn.answer.usage.total_tokens} tokens</span>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="mt-4 flex items-end gap-2">
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              ask(question);
            }
          }}
          rows={1}
          placeholder="Ask about a policy, deadline or limit…"
          className="min-h-[44px] flex-1 resize-none rounded-xl border border-edge bg-white/[0.03] px-3 py-2.5 text-sm text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-accent/70"
        />
        <button
          type="submit"
          disabled={busy || question.trim().length < 2}
          className="rounded-xl bg-accent px-4 py-2.5 text-sm font-medium text-ink transition hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "Thinking…" : "Ask"}
        </button>
      </form>
    </section>
  );
}
