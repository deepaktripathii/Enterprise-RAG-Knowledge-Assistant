# RAG Knowledge Assistant frontend

Next.js 14 + React 18 + Tailwind CSS client for the RAG Knowledge Assistant API.

```bash
cd frontend
npm install
cp .env.local.example .env.local      # adjust NEXT_PUBLIC_API_URL if needed
npm run dev                            # http://localhost:3000
```

Production build:

```bash
npm run build
npm start
```

The UI has three parts:

1. **Documents** - drag-and-drop PDF upload, live processing status
   (pending / processing / completed / failed), page and chunk counts, retry and
   delete actions.
2. **Chat** - grounded answers with inline citation cards (document + page +
   snippet), conversation continuity, and a "grounded / fallback" indicator.
3. **Operations** - document, vector and query counters, latency, fallback rate
   and the latest retrieval evaluation report with a one-click re-run.
