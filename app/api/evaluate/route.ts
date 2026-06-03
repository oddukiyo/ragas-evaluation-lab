import { NextResponse } from "next/server";

import { PAPER_KNOWLEDGE } from "@/lib/paper-knowledge";
import { callOpenRouter } from "@/lib/openrouter";
import {
  cosineSimilarity,
  retrieveRelevantChunks,
  round,
  splitSentences,
  tokenize,
  type RetrievedChunk,
} from "@/lib/rag";

type RequestBody = {
  apiKey?: unknown;
  model?: unknown;
  question?: unknown;
  topK?: unknown;
};

type LocalEvaluation = {
  faithfulness: number;
  answerRelevance: number;
  contextRelevance: number;
  report: string;
  details: {
    statements: string[];
    statementChecks: {
      statement: string;
      supported: boolean;
      bestContextSimilarity: number;
      reason: string;
    }[];
    generatedQuestions: string[];
    answerQuestionSimilarities: number[];
    relevantSentences: string[];
    totalContextSentences: number;
  };
};

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const input = validateBody(body);

    const cleanQuestion = removeDiacritics(input.question);

    const retrieval = retrieveRelevantChunks({
      knowledgeBase: PAPER_KNOWLEDGE,
      question: cleanQuestion,
      topK: input.topK,
    });

    const context = retrieval.retrieved
      .map((chunk) => chunk.text)
      .join("\n\n---\n\n");

    const answerPrompt = `
You are a RAG assistant for a student project about the RAGAS paper.
Answer the question in Persian.
Do not use Arabic/Persian diacritics or vowel marks such as َ ِ ُ ّ ْ.
Use only the retrieved context below.
If the context is insufficient, say that the retrieved context is insufficient.

Retrieved context:
${context}

Question:
${cleanQuestion}
`.trim();

    const answerOutput = await callOpenRouter({
      apiKey: input.apiKey,
      model: input.model,
      prompt: answerPrompt,
      temperature: 0,
    });

    const cleanAnswer = removeDiacritics(answerOutput.content);

    const evaluation = evaluateRagasLocally({
      question: cleanQuestion,
      answer: cleanAnswer,
      context,
      retrievedChunks: retrieval.retrieved,
    });

    return NextResponse.json({
      question: cleanQuestion,
      answer: cleanAnswer,
      model: answerOutput.model,
      retrievedChunks: retrieval.retrieved,
      stats: retrieval.stats,
      evaluation,
    });
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error ? error.message : "Unexpected server error.",
      },
      { status: 500 }
    );
  }
}

function validateBody(body: RequestBody) {
  const apiKey = typeof body.apiKey === "string" ? body.apiKey.trim() : "";
  const question = typeof body.question === "string" ? body.question.trim() : "";
  const model = typeof body.model === "string" ? body.model.trim() : "";
  const topK = typeof body.topK === "number" ? body.topK : 3;

  if (!apiKey) {
    throw new Error("OpenRouter API key is required.");
  }

  if (!question) {
    throw new Error("Question is required.");
  }

  return {
    apiKey,
    question,
    model: model || "openrouter/free",
    topK: Math.min(Math.max(topK, 1), 5),
  };
}

function evaluateRagasLocally({
  question,
  answer,
  context,
  retrievedChunks,
}: {
  question: string;
  answer: string;
  context: string;
  retrievedChunks: RetrievedChunk[];
}): LocalEvaluation {
  const answerSentences = splitSentences(answer);
  const contextSentences = splitSentences(context);

  const statementChecks = answerSentences.map((statement) => {
    const bestContextSimilarity =
      contextSentences.length === 0
        ? 0
        : Math.max(
            ...contextSentences.map((contextSentence) =>
              textSimilarity(statement, contextSentence)
            )
          );

    const supported = bestContextSimilarity >= 0.05;

    return {
      statement,
      supported,
      bestContextSimilarity: round(bestContextSimilarity),
      reason: supported
        ? "This statement has lexical overlap with the retrieved context."
        : "This statement has weak overlap with the retrieved context.",
    };
  });

  const supportedCount = statementChecks.filter((item) => item.supported).length;

  const faithfulness =
    statementChecks.length === 0 ? 0 : supportedCount / statementChecks.length;

  const answerRelevance = textSimilarity(question, answer);

  const averageRetrievalScore =
    retrievedChunks.length === 0
      ? 0
      : retrievedChunks.reduce((sum, chunk) => sum + chunk.score, 0) /
        retrievedChunks.length;

  const contextRelevance = Math.min(1, averageRetrievalScore * 2.5);

  const generatedQuestions = [
    question,
    `What information does the answer provide about ${extractMainKeyword(
      question
    )}?`,
    `How does the answer relate to ${extractMainKeyword(question)}?`,
  ];

  const answerQuestionSimilarities = generatedQuestions.map((item) =>
    round(textSimilarity(item, answer))
  );

  const relevantSentences = contextSentences.filter((sentence) => {
    return textSimilarity(question, sentence) >= 0.04;
  });

  const scores = {
    faithfulness: round(faithfulness),
    answerRelevance: round(answerRelevance),
    contextRelevance: round(contextRelevance),
  };

  return {
    ...scores,
    report: buildReport(scores),
    details: {
      statements: answerSentences,
      statementChecks,
      generatedQuestions,
      answerQuestionSimilarities,
      relevantSentences,
      totalContextSentences: contextSentences.length,
    },
  };
}

function textSimilarity(a: string, b: string) {
  const aTokens = usefulTokens(a);
  const bTokens = usefulTokens(b);

  if (aTokens.length === 0 || bTokens.length === 0) {
    return 0;
  }

  const vocabulary = Array.from(new Set([...aTokens, ...bTokens]));

  const aVector = vocabulary.map((term) => {
    return aTokens.filter((token) => token === term).length;
  });

  const bVector = vocabulary.map((term) => {
    return bTokens.filter((token) => token === term).length;
  });

  const cosine = cosineSimilarity(aVector, bVector);

  const aSet = new Set(aTokens);
  const bSet = new Set(bTokens);

  const intersection = Array.from(aSet).filter((token) => {
    return bSet.has(token);
  }).length;

  const overlap = intersection / Math.min(aSet.size, bSet.size);

  return Math.min(1, (cosine + overlap) / 2);
}

function usefulTokens(text: string) {
  return tokenize(text).filter((token) => {
    return token.length > 1 && !/^[.,!?؟؛:]$/.test(token);
  });
}

function extractMainKeyword(question: string) {
  const tokens = usefulTokens(question);
  return tokens[0] || "the topic";
}

function buildReport(scores: {
  faithfulness: number;
  answerRelevance: number;
  contextRelevance: number;
}) {
  const comments: string[] = [];

  comments.push(
    scores.faithfulness >= 0.6
      ? "The answer is mostly grounded in the retrieved evidence."
      : "Some parts of the answer may not be strongly grounded in the retrieved evidence."
  );

  comments.push(
    scores.answerRelevance >= 0.4
      ? "The answer is reasonably related to the original question."
      : "The answer may not fully address the original question."
  );

  comments.push(
    scores.contextRelevance >= 0.4
      ? "The retrieved context is relatively focused."
      : "The retrieved context may include extra or weakly relevant information."
  );

  return comments.join(" ");
}

function removeDiacritics(text: string) {
  return text.replace(/[\u064B-\u065F\u0670\u06D6-\u06ED]/g, "");
}
