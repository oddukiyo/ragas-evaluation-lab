export type Chunk = {
  id: string;
  text: string;
  tokens: string[];
};

export type RetrievedChunk = Chunk & {
  score: number;
};

export type PipelineStats = {
  tokenCount: number;
  chunkCount: number;
  vectorDimension: number;
  topK: number;
  tokenizer: string;
  chunking: string;
  vectorizer: string;
  similarity: string;
};

export function normalizeText(text: string) {
  return text.replace(/\s+/g, " ").trim();
}

export function tokenize(text: string): string[] {
  return (
    normalizeText(text)
      .toLowerCase()
      .match(/[\p{L}\p{N}_]+|[.,!?؟؛:]/gu)
      ?.filter(Boolean) ?? []
  );
}

export function chunkText(
  text: string,
  chunkSize = 75,
  overlap = 18
): Chunk[] {
  const tokens = tokenize(text);
  const chunks: Chunk[] = [];

  let start = 0;
  let index = 0;

  while (start < tokens.length) {
    const end = Math.min(start + chunkSize, tokens.length);
    const chunkTokens = tokens.slice(start, end);

    chunks.push({
      id: `chunk-${index + 1}`,
      text: chunkTokens.join(" "),
      tokens: chunkTokens,
    });

    if (end === tokens.length) break;

    start = Math.max(0, end - overlap);
    index += 1;
  }

  return chunks;
}

export function buildVocabulary(chunks: Chunk[], query: string): string[] {
  const vocabulary = new Set<string>();

  for (const chunk of chunks) {
    for (const token of chunk.tokens) {
      if (isUsefulToken(token)) {
        vocabulary.add(token);
      }
    }
  }

  for (const token of tokenize(query)) {
    if (isUsefulToken(token)) {
      vocabulary.add(token);
    }
  }

  return Array.from(vocabulary).sort();
}

export function vectorizeTfIdf(
  textTokens: string[],
  allDocuments: string[][],
  vocabulary: string[]
) {
  return vocabulary.map(
    (term) => termFrequency(term, textTokens) * inverseDocumentFrequency(term, allDocuments)
  );
}

export function retrieveRelevantChunks({
  knowledgeBase,
  question,
  topK = 3,
}: {
  knowledgeBase: string;
  question: string;
  topK?: number;
}) {
  const chunks = chunkText(knowledgeBase);
  const vocabulary = buildVocabulary(chunks, question);
  const allDocs = chunks.map((chunk) => chunk.tokens);

  const queryVector = vectorizeTfIdf(tokenize(question), allDocs, vocabulary);

  const retrieved = chunks
    .map((chunk) => {
      const chunkVector = vectorizeTfIdf(chunk.tokens, allDocs, vocabulary);

      return {
        ...chunk,
        score: round(cosineSimilarity(queryVector, chunkVector)),
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  const stats: PipelineStats = {
    tokenCount: tokenize(knowledgeBase).length,
    chunkCount: chunks.length,
    vectorDimension: vocabulary.length,
    topK,
    tokenizer: "Regex tokenizer for Persian/English tokens",
    chunking: "Token-based chunking with overlap",
    vectorizer: "TF-IDF word vectorization",
    similarity: "Cosine similarity",
  };

  return {
    retrieved,
    chunks,
    vocabulary,
    stats,
  };
}

export function cosineSimilarity(a: number[], b: number[]) {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) return 0;

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function splitSentences(text: string) {
  return normalizeText(text)
    .split(/(?<=[.!?؟])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}

export function round(value: number) {
  return Math.round(value * 1000) / 1000;
}

function termFrequency(term: string, tokens: string[]) {
  if (tokens.length === 0) return 0;

  return tokens.filter((token) => token === term).length / tokens.length;
}

function inverseDocumentFrequency(term: string, docs: string[][]) {
  const total = docs.length;
  const containing = docs.filter((doc) => doc.includes(term)).length;

  return Math.log((total + 1) / (containing + 1)) + 1;
}

function isUsefulToken(token: string) {
  return token.length > 1 && !/^[.,!?؟؛:]$/.test(token);
}
