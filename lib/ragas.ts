import { callOpenRouter, parseJsonFromText } from "./openrouter";
import { cosineSimilarity, splitSentences, tokenize, round } from "./rag";

export type StatementCheck = {
  statement: string;
  supported: boolean;
  reason: string;
};

export type RagasEvaluation = {
  faithfulness: number;
  answerRelevance: number;
  contextRelevance: number;
  report: string;
  details: {
    statements: string[];
    statementChecks: StatementCheck[];
    generatedQuestions: string[];
    answerQuestionSimilarities: number[];
    relevantSentences: string[];
    totalContextSentences: number;
  };
};

export async function evaluateRagas({
  apiKey,
  model,
  question,
  answer,
  context,
}: {
  apiKey: string;
  model: string;
  question: string;
  answer: string;
  context: string;
}): Promise<RagasEvaluation> {
  const statements = await extractStatements({
    apiKey,
    model,
    question,
    answer,
  });

  const statementChecks = await checkStatements({
    apiKey,
    model,
    context,
    statements,
  });

  const generatedQuestions = await generateQuestionsFromAnswer({
    apiKey,
    model,
    answer,
  });

  const answerQuestionSimilarities = generatedQuestions.map((generatedQuestion) =>
    round(tokenCosineSimilarity(question, generatedQuestion))
  );

  const relevantSentences = await extractRelevantSentences({
    apiKey,
    model,
    question,
    context,
  });

  const totalContextSentences = splitSentences(context).length;

  const faithfulness =
    statements.length === 0
      ? 0
      : statementChecks.filter((item) => item.supported).length / statements.length;

  const answerRelevance =
    answerQuestionSimilarities.length === 0
      ? 0
      : mean(answerQuestionSimilarities);

  const contextRelevance =
    totalContextSentences === 0
      ? 0
      : relevantSentences.length / totalContextSentences;

  const scores = {
    faithfulness: round(faithfulness),
    answerRelevance: round(answerRelevance),
    contextRelevance: round(contextRelevance),
  };

  return {
    ...scores,
    report: buildReport(scores),
    details: {
      statements,
      statementChecks,
      generatedQuestions,
      answerQuestionSimilarities,
      relevantSentences,
      totalContextSentences,
    },
  };
}

async function extractStatements({
  apiKey,
  model,
  question,
  answer,
}: {
  apiKey: string;
  model: string;
  question: string;
  answer: string;
}) {
  const prompt = `You are implementing the Faithfulness step from the RAGAS paper. Extract short factual statements from the answer. Return ONLY valid JSON. Format: {"statements":["statement 1","statement 2"]}

Question:
${question}

Answer:
${answer}`;

  const output = await callOpenRouter({ apiKey, model, prompt });
  const json = parseJsonFromText(output.content);

  return asStringArray(json.statements);
}

async function checkStatements({
  apiKey,
  model,
  context,
  statements,
}: {
  apiKey: string;
  model: string;
  context: string;
  statements: string[];
}): Promise<StatementCheck[]> {
  if (statements.length === 0) return [];

  const prompt = `Check whether each statement is supported by the retrieved context. Return ONLY valid JSON. Format: {"checks":[{"statement":"statement text","supported":true,"reason":"short reason"}]}

Retrieved context:
${context}

Statements:
${JSON.stringify(statements)}`;

  const output = await callOpenRouter({ apiKey, model, prompt });
  const json = parseJsonFromText(output.content);

  if (!Array.isArray(json.checks)) return [];

  return json.checks.map((item: Record<string, unknown>) => ({
    statement: typeof item.statement === "string" ? item.statement : "",
    supported: Boolean(item.supported),
    reason: typeof item.reason === "string" ? item.reason : "",
  }));
}

async function generateQuestionsFromAnswer({
  apiKey,
  model,
  answer,
}: {
  apiKey: string;
  model: string;
  answer: string;
}) {
  const prompt = `You are implementing the Answer Relevance step from the RAGAS paper. Generate 3 different questions that could be answered by the answer. Return ONLY valid JSON. Format: {"questions":["question 1","question 2","question 3"]}

Answer:
${answer}`;

  const output = await callOpenRouter({ apiKey, model, prompt });
  const json = parseJsonFromText(output.content);

  return asStringArray(json.questions);
}

async function extractRelevantSentences({
  apiKey,
  model,
  question,
  context,
}: {
  apiKey: string;
  model: string;
  question: string;
  context: string;
}) {
  const prompt = `You are implementing the Context Relevance step from the RAGAS paper. Extract only the exact context sentences that are necessary to answer the question. Do not rewrite sentences. Return ONLY valid JSON. Format: {"relevantSentences":["sentence 1","sentence 2"]}

Question:
${question}

Context:
${context}`;

  const output = await callOpenRouter({ apiKey, model, prompt });
  const json = parseJsonFromText(output.content);

  return asStringArray(json.relevantSentences);
}

function tokenCosineSimilarity(a: string, b: string) {
  const aTokens = tokenize(a);
  const bTokens = tokenize(b);
  const vocabulary = Array.from(new Set([...aTokens, ...bTokens]));

  const aVector = vocabulary.map(
    (token) => aTokens.filter((item) => item === token).length
  );

  const bVector = vocabulary.map(
    (token) => bTokens.filter((item) => item === token).length
  );

  return cosineSimilarity(aVector, bVector);
}

function mean(values: number[]) {
  if (values.length === 0) return 0;

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];

  return value.filter((item): item is string => typeof item === "string");
}

function buildReport(scores: {
  faithfulness: number;
  answerRelevance: number;
  contextRelevance: number;
}) {
  const comments: string[] = [];

  comments.push(
    scores.faithfulness >= 0.75
      ? "The answer is strongly grounded in the retrieved context."
      : "Some claims may not be fully supported by the retrieved context."
  );

  comments.push(
    scores.answerRelevance >= 0.6
      ? "The answer is reasonably aligned with the original question."
      : "The answer may not directly address the original question."
  );

  comments.push(
    scores.contextRelevance >= 0.45
      ? "The retrieved evidence is fairly focused."
      : "The retrieved context may contain extra or weakly relevant information."
  );

  return comments.join(" ");
}
