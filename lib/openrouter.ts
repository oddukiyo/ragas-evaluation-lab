const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

export async function callOpenRouter({
  apiKey,
  model,
  prompt,
  temperature = 0,
}: {
  apiKey: string;
  model: string;
  prompt: string;
  temperature?: number;
}) {
  const response = await fetch(OPENROUTER_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "https://vercel.app",
      "X-Title": "RAGAS Evaluation Lab",
    },
    body: JSON.stringify({
      model,
      messages: [
        {
          role: "user",
          content: prompt,
        },
      ],
      temperature,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenRouter API error: ${errorText}`);
  }

  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content;
  const usedModel = data?.model || model;

  if (typeof content !== "string") {
    throw new Error("OpenRouter returned an empty response.");
  }

  return {
    content: content.trim(),
    model: usedModel,
  };
}

export function parseJsonFromText(text: string) {
  const cleaned = text
    .trim()
    .replace(/^```json/i, "")
    .replace(/^```/i, "")
    .replace(/```$/i, "")
    .trim();

  try {
    return JSON.parse(cleaned);
  } catch {
    const match = cleaned.match(/\{[\s\S]*\}/);

    if (!match) {
      throw new Error(`Could not parse JSON from model output: ${text}`);
    }

    return JSON.parse(match[0]);
  }
}
