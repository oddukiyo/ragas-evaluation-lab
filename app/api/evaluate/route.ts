import {NextResponse} from 'next/server'
import {PAPER_KNOWLEDGE} from '@/lib/paper-knowledge'
import {callOpenRouter} from '@/lib/openrouter'
import {retrieveRelevantChunks} from '@/lib/rag'
import {evaluateRagas} from '@/lib/ragas'
type RequestBody={apiKey?:unknown;model?:unknown;question?:unknown;topK?:unknown}
export async function POST(request:Request){try{const body=await request.json();const input=validateBody(body);const retrieval=retrieveRelevantChunks({knowledgeBase:PAPER_KNOWLEDGE,question:input.question,topK:input.topK});const context=retrieval.retrieved.map(c=>c.text).join('\n\n---\n\n');const answerPrompt=`You are a RAG assistant for a student project about the RAGAS paper. Answer in Persian. Use only the retrieved context below. If the context is insufficient, say that the retrieved context is insufficient.

Retrieved context:
${context}

Question:
${input.question}`;const answerOutput=await callOpenRouter({apiKey:input.apiKey,model:input.model,prompt:answerPrompt,temperature:0});const evaluation=await evaluateRagas({apiKey:input.apiKey,model:input.model,question:input.question,answer:answerOutput.content,context});return NextResponse.json({question:input.question,answer:answerOutput.content,model:answerOutput.model,retrievedChunks:retrieval.retrieved,stats:retrieval.stats,evaluation})}catch(error){return NextResponse.json({error:error instanceof Error?error.message:'Unexpected server error.'},{status:500})}}
function validateBody(body:RequestBody){const apiKey=typeof body.apiKey==='string'?body.apiKey.trim():'';const question=typeof body.question==='string'?body.question.trim():'';const model=typeof body.model==='string'?body.model.trim():'';const topK=typeof body.topK==='number'?body.topK:3;if(!apiKey)throw new Error('OpenRouter API key is required.');if(!question)throw new Error('Question is required.');return{apiKey,question,model:model||'openrouter/free',topK:Math.min(Math.max(topK,1),5)}}
