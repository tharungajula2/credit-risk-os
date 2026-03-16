import { createAgentUIStreamResponse, ToolLoopAgent } from 'ai';
import { google } from '@ai-sdk/google';
import fs from 'fs';
import path from 'path';

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();

    // 1. Safely load the RAG Context
    let ragContext = "";
    try {
      const brainDir = path.join(process.cwd(), 'brain');
      if (fs.existsSync(brainDir)) {
        const files = fs.readdirSync(brainDir);
        for (const file of files) {
          if (file.endsWith('.md')) {
            const content = fs.readFileSync(path.join(brainDir, file), 'utf-8');
            ragContext += `\n--- Document: ${file} ---\n${content}\n`;
          }
        }
      }

      const portfolioDir = path.join(process.cwd(), 'brain', 'portfolio');
      if (fs.existsSync(portfolioDir)) {
        const portFiles = fs.readdirSync(portfolioDir);
        for (const file of portFiles) {
          if (file.endsWith('.md')) {
            const content = fs.readFileSync(path.join(portfolioDir, file), 'utf-8');
            ragContext += `\n--- Portfolio: ${file} ---\n${content}\n`;
          }
        }
      }
    } catch (readError) {
      console.error("Failed to read brain folder:", readError);
      ragContext = "Knowledge base currently unavailable.";
    }

    const systemPrompt = `You are Vian AI, the AI assistant for Tharun's Quant OS. You are a highly restricted, zero-hallucination RAG agent. I will provide you with Tharun's personal knowledge base. You must answer the user's questions strictly and exclusively using the information provided in the knowledge base below. If the user asks a question that is not covered in the knowledge base, you must politely reply: 'I am highly restricted to Tharun's Quant OS knowledge base. I cannot answer outside queries.' Do not make up information. Here is the knowledge base:\n\n${ragContext}`;

    const crosAgent = new ToolLoopAgent({
      model: google('gemini-2.5-flash-lite'),
      instructions: systemPrompt,
    });

    return createAgentUIStreamResponse({
      agent: crosAgent,
      uiMessages: messages,
    });

  } catch (error) {
    console.error("Fatal API Error:", error);
    return new Response(JSON.stringify({ error: "Internal Server Error" }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}