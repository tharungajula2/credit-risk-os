import { createAgentUIStreamResponse, ToolLoopAgent } from 'ai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import fs from 'fs';
import path from 'path';

// Force load the .env.local file directly in case the server hasn't been restarted
let apiKey = process.env.GOOGLE_GENERATIVE_AI_API_KEY;
try {
  const envPath = path.join(process.cwd(), '.env.local');
  if (fs.existsSync(envPath)) {
    const envContent = fs.readFileSync(envPath, 'utf-8');
    const match = envContent.match(/GOOGLE_GENERATIVE_AI_API_KEY=(.+)/);
    if (match && match[1]) apiKey = match[1].trim();
  }
} catch (e) {
  console.log("Failed to load .env.local dynamically");
}

const googleExt = createGoogleGenerativeAI({
  apiKey: apiKey || '',
});

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
      model: googleExt('gemini-2.5-flash-lite'),
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