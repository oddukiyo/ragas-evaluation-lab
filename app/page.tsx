"use client"

import { useMemo, useState } from "react"
import {
  Activity,
  BarChart3,
  Brain,
  FileSearch,
  KeyRound,
  Layers3,
  MessageSquareText,
  Network,
  ShieldCheck,
} from "lucide-react"

type RetrievedChunk = {
  id: string
  text: string
  score: number
}

type Evaluation = {
  faithfulness: number
  answerRelevance: number
  contextRelevance: number
  report: string
  details: Record<string, unknown>
}

type ApiResult = {
  question: string
  answer: string
  model: string
  retrievedChunks: RetrievedChunk[]
  stats: {
    tokenCount: number
    chunkCount: number
    vectorDimension: number
    topK: number
    tokenizer: string
    chunking: string
    vectorizer: string
    similarity: string
  }
  evaluation: Evaluation
}

const sampleQuestions = [
  "RAGAS چیست و چرا برای سیستم‌های RAG مهم است؟",
  "Faithfulness در مقاله RAGAS چگونه محاسبه می‌شود؟",
  "تفاوت Answer Relevance و Context Relevance چیست؟",
  "محدودیت‌های RAGAS چیست؟",
]

export default function Home() {
  const [apiKey, setApiKey] = useState("")
  const [model, setModel] = useState("openrouter/free")
  const [question, setQuestion] = useState(sampleQuestions[0])
  const [topK, setTopK] = useState(3)
  const [result, setResult] = useState<ApiResult | null>(null)
  const [error, setError] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  const canSubmit = useMemo(() => {
    return apiKey.trim().length > 0 && question.trim().length > 0
  }, [apiKey, question])

  async function runEvaluation() {
    setIsLoading(true)
    setError("")
    setResult(null)

    try {
      const response = await fetch("/api/evaluate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          apiKey,
          model,
          question,
          topK,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Evaluation failed.")
      }

      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <div className="hero-grid">
          <div>
            <span className="badge">
              <Brain size={16} />
              RAGAS Paper Reproduction Demo
            </span>

            <h1>RAGAS Evaluation Lab</h1>

            <p>
              یک داشبورد مستقل برای نمایش مسیر کامل RAG: از tokenization و
              chunking تا vectorization، retrieval، تولید پاسخ با API و ارزیابی
              سه معیار اصلی مقاله RAGAS.
            </p>
          </div>

          <div className="pipeline-card">
            <PipelineStep
              icon={<Layers3 size={16} />}
              text="Tokenization + Chunking"
            />
            <PipelineStep
              icon={<Network size={16} />}
              text="TF-IDF Word Vectorization"
            />
            <PipelineStep
              icon={<FileSearch size={16} />}
              text="Cosine Similarity Retrieval"
            />
            <PipelineStep
              icon={<MessageSquareText size={16} />}
              text="API-based Answer Generation"
            />
            <PipelineStep
              icon={<BarChart3 size={16} />}
              text="RAGAS-style Evaluation"
            />
          </div>
        </div>
      </section>

      <section className="main-grid">
        <div className="card">
          <h2 className="card-title">
            <span>ورودی آزمایش</span>
            <KeyRound size={20} />
          </h2>

          <label className="label" htmlFor="api-key">
            OpenRouter API Key
          </label>
          <input
            id="api-key"
            className="input"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
            placeholder="sk-or-..."
            type="password"
          />

          <div className="form-row">
            <div>
              <label className="label" htmlFor="model">
                مدل API
              </label>
              <input
                id="model"
                className="input"
                value={model}
                onChange={(event) => setModel(event.target.value)}
                placeholder="openrouter/free"
              />
            </div>

            <div>
              <label className="label" htmlFor="top-k">
                Top-K Contexts
              </label>
              <select
                id="top-k"
                className="select"
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              >
                {[1, 2, 3, 4, 5].map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <label className="label" htmlFor="question">
            سؤال
          </label>
          <textarea
            id="question"
            className="textarea"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
          />

          <button
            className="primary-button"
            onClick={runEvaluation}
            disabled={!canSubmit || isLoading}
          >
            {isLoading ? "در حال اجرای pipeline..." : "اجرای RAGAS Pipeline"}
          </button>

          <div className="sample-grid">
            {sampleQuestions.map((sample) => (
              <button
                key={sample}
                className="sample-button"
                onClick={() => setQuestion(sample)}
                type="button"
              >
                {sample}
              </button>
            ))}
          </div>
        </div>

        <div className="result-stack">
          {error ? <div className="error">{error}</div> : null}

          <div className="card">
            <h2 className="card-title">
              <span>پاسخ تولیدشده</span>
              <MessageSquareText size={20} />
            </h2>

            <div className="answer-box">
              {isLoading
                ? "در حال بازیابی context، تولید پاسخ و محاسبه معیارها..."
                : result?.answer ||
                  "بعد از اجرای pipeline، پاسخ مدل در این بخش نمایش داده می‌شود."}
            </div>
          </div>

          <MetricPanel evaluation={result?.evaluation} />
          <TechnicalPanel stats={result?.stats} />
          <ContextPanel chunks={result?.retrievedChunks ?? []} />

          {result?.evaluation ? (
            <div className="card">
              <h2 className="card-title">
                <span>گزارش تحلیلی</span>
                <ShieldCheck size={20} />
              </h2>

              <div className="answer-box">{result.evaluation.report}</div>

              <details className="details">
                <summary>جزئیات خام محاسبات RAGAS</summary>
                <pre>{JSON.stringify(result.evaluation.details, null, 2)}</pre>
              </details>
            </div>
          ) : null}
        </div>
      </section>

      <p className="footer-note">
        این پروژه از پایه برای نمایش مقاله RAGAS ساخته شده و شامل tokenization،
        chunking، TF-IDF vectorization، retrieval و RAGAS-style scoring است.
      </p>
    </main>
  )
}

function PipelineStep({
  icon,
  text,
}: {
  icon: React.ReactNode
  text: string
}) {
  return (
    <div className="pipeline-step">
      <span className="dot" />
      {icon}
      <span>{text}</span>
    </div>
  )
}

function MetricPanel({ evaluation }: { evaluation?: Evaluation }) {
  return (
    <div className="card">
      <h2 className="card-title">
        <span>RAGAS Metrics</span>
        <Activity size={20} />
      </h2>

      <div className="metrics-grid">
        <Metric
          title="Faithfulness"
          value={evaluation?.faithfulness ?? 0}
          description="آیا ادعاهای پاسخ توسط context پشتیبانی می‌شوند؟"
        />
        <Metric
          title="Answer Relevance"
          value={evaluation?.answerRelevance ?? 0}
          description="آیا پاسخ مستقیماً به سؤال اصلی مربوط است؟"
        />
        <Metric
          title="Context Relevance"
          value={evaluation?.contextRelevance ?? 0}
          description="آیا context بازیابی‌شده متمرکز و مفید است؟"
        />
      </div>
    </div>
  )
}

function Metric({
  title,
  value,
  description,
}: {
  title: string
  value: number
  description: string
}) {
  const percentage = Math.round(value * 100)

  return (
    <div className="metric">
      <div className="metric-header">
        <span className="metric-name">{title}</span>
        <span>{percentage}%</span>
      </div>

      <div className="metric-value">{value.toFixed(2)}</div>

      <div className="bar">
        <div className="bar-fill" style={{ width: `${percentage}%` }} />
      </div>

      <p className="metric-desc">{description}</p>
    </div>
  )
}

function TechnicalPanel({ stats }: { stats?: ApiResult["stats"] }) {
  return (
    <div className="card">
      <h2 className="card-title">
        <span>Technical Pipeline</span>
        <Network size={20} />
      </h2>

      <div className="tech-grid">
        <TechBox value={stats?.tokenCount ?? "-"} label="Token count" />
        <TechBox value={stats?.chunkCount ?? "-"} label="Chunks" />
        <TechBox
          value={stats?.vectorDimension ?? "-"}
          label="Vector dimension"
        />
        <TechBox value={stats?.topK ?? "-"} label="Top-K retrieval" />
        <TechBox
          value={stats?.vectorizer ?? "TF-IDF"}
          label="Embedding / Vectorizer"
        />
        <TechBox value={stats?.similarity ?? "Cosine"} label="Similarity" />
      </div>
    </div>
  )
}

function TechBox({ value, label }: { value: number | string; label: string }) {
  return (
    <div className="tech-box">
      <span>{value}</span>
      <p>{label}</p>
    </div>
  )
}

function ContextPanel({ chunks }: { chunks: RetrievedChunk[] }) {
  return (
    <div className="card">
      <h2 className="card-title">
        <span>Retrieved Evidence</span>
        <FileSearch size={20} />
      </h2>

      <div className="context-list">
        {chunks.length === 0 ? (
          <div className="context-item">
            <div className="context-text">
              بعد از اجرای سیستم، chunkهای بازیابی‌شده در این بخش نمایش داده
              می‌شوند.
            </div>
          </div>
        ) : (
          chunks.map((chunk) => (
            <div className="context-item" key={chunk.id}>
              <div className="context-meta">
                <span>{chunk.id}</span>
                <span>similarity: {chunk.score.toFixed(3)}</span>
              </div>
              <div className="context-text">{chunk.text}</div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}