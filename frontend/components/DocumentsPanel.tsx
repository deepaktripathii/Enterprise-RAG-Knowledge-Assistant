"use client";

import { useRef, useState } from "react";
import { api, errorMessage } from "@/lib/api";
import type { DocumentRecord } from "@/lib/types";

const STATUS_STYLE: Record<string, string> = {
  completed: "bg-mint/10 text-mint border border-mint/30",
  processing: "bg-accent/10 text-accent border border-accent/30",
  pending: "bg-amber-400/10 text-amber-300 border border-amber-400/30",
  failed: "bg-rose-500/10 text-rose-300 border border-rose-500/30",
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function DocumentsPanel({
  documents,
  onChanged,
}: {
  documents: DocumentRecord[];
  onChanged: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);

  async function upload(files: FileList | null) {
    if (!files || files.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      for (const file of Array.from(files)) {
        await api.upload(file);
      }
      onChanged();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  async function remove(id: string) {
    setActiveId(id);
    try {
      await api.deleteDocument(id);
      onChanged();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setActiveId(null);
    }
  }

  async function retry(id: string) {
    setActiveId(id);
    try {
      await api.processDocument(id);
      onChanged();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setActiveId(null);
    }
  }

  const totalChunks = documents.reduce((sum, doc) => sum + doc.chunk_count, 0);

  return (
    <section className="card flex h-full flex-col p-5">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-white">Knowledge base</h2>
          <p className="text-sm text-slate-400">
            {documents.length} document{documents.length === 1 ? "" : "s"} · {totalChunks} chunks
          </p>
        </div>
        <span className="pill bg-white/5 text-slate-300">PDF</span>
      </header>

      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          upload(event.dataTransfer.files);
        }}
        className={`mt-4 rounded-xl border border-dashed p-4 text-center transition ${
          dragging
            ? "border-accent bg-accent/10"
            : "border-edge bg-white/[0.02] hover:border-accent/60"
        }`}
      >
        <p className="text-sm text-slate-300">
          Drop a PDF here or{" "}
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="font-medium text-accent hover:underline"
          >
            browse
          </button>
        </p>
        <p className="mt-1 text-xs text-slate-500">
          Text-based PDFs up to 20 MB. Scanned files need OCR first.
        </p>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          multiple
          hidden
          onChange={(event) => upload(event.target.files)}
        />
        {uploading && <p className="mt-2 text-xs text-accent">Uploading and indexing…</p>}
      </div>

      {error && (
        <p className="mt-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
          {error}
        </p>
      )}

      <div className="mt-4 flex-1 space-y-2 overflow-y-auto pr-1">
        {documents.length === 0 && (
          <p className="rounded-lg border border-edge bg-white/[0.02] px-3 py-6 text-center text-sm text-slate-500">
            No documents yet. Upload a policy PDF to get started.
          </p>
        )}

        {documents.map((document) => (
          <article
            key={document.id}
            className="rounded-xl border border-edge bg-white/[0.02] p-3"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-slate-100" title={document.filename}>
                  {document.filename}
                </p>
                <p className="mt-0.5 text-xs text-slate-500">
                  {formatBytes(document.size_bytes)}
                  {document.page_count > 0 && ` · ${document.page_count} pages`}
                  {document.chunk_count > 0 && ` · ${document.chunk_count} chunks`}
                </p>
              </div>
              <span className={`pill ${STATUS_STYLE[document.status] ?? "bg-white/5"}`}>
                {document.status}
              </span>
            </div>

            {document.status === "failed" && document.error_message && (
              <p className="mt-2 rounded-md bg-rose-500/10 px-2 py-1 text-xs text-rose-200">
                {document.error_message}
              </p>
            )}

            <div className="mt-2 flex items-center gap-3 text-xs">
              {document.status === "failed" && (
                <button
                  type="button"
                  onClick={() => retry(document.id)}
                  disabled={activeId === document.id}
                  className="text-accent hover:underline disabled:opacity-50"
                >
                  Retry indexing
                </button>
              )}
              <button
                type="button"
                onClick={() => remove(document.id)}
                disabled={activeId === document.id}
                className="text-slate-500 transition hover:text-rose-300 disabled:opacity-50"
              >
                Delete
              </button>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
