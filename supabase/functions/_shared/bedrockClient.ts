// AWS Bedrock client using AWS SDK - simple and straightforward like Anthropic client
import { BedrockRuntimeClient, InvokeModelCommand, InvokeModelWithResponseStreamCommand } from 'npm:@aws-sdk/client-bedrock-runtime@3.645.0';

const AWS_REGION = Deno.env.get('AWS_REGION') || 'us-east-1';
const AWS_ACCESS_KEY_ID = Deno.env.get('AWS_ACCESS_KEY_ID') ?? '';
const AWS_SECRET_ACCESS_KEY = Deno.env.get('AWS_SECRET_ACCESS_KEY') ?? '';

// Initialize Bedrock client
const bedrockClient = new BedrockRuntimeClient({
  region: AWS_REGION,
  credentials: {
    accessKeyId: AWS_ACCESS_KEY_ID,
    secretAccessKey: AWS_SECRET_ACCESS_KEY,
  },
});

export interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content:
    | string
    | Array<{ type: string; text?: string; image_url?: { url: string } }>;
  tool_call_id?: string;
  tool_calls?: Array<{
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
  }>;
}

// Convert OpenAI messages to Bedrock format
function convertMessages(
  openAIMessages: OpenAIMessage[],
): { system?: string; messages: Array<{ role: string; content: any }> } {
  let system: string | undefined;
  const messages: Array<{ role: string; content: any }> = [];

  for (const msg of openAIMessages) {
    if (msg.role === 'system') {
      system = typeof msg.content === 'string' ? msg.content : '';
      continue;
    }

    if (msg.role === 'tool') {
      continue;
    }

    let content: any;

    if (typeof msg.content === 'string') {
      content = msg.content;
    } else if (Array.isArray(msg.content)) {
      content = msg.content.map((part) => {
        if (part.type === 'text') {
          return { type: 'text', text: part.text || '' };
        } else if (part.type === 'image_url' && part.image_url) {
          const url = part.image_url.url;
          if (url.startsWith('data:')) {
            const match = url.match(/^data:([^;]+);base64,(.+)$/);
            if (match) {
              return {
                type: 'image',
                source: {
                  type: 'base64',
                  media_type: match[1],
                  data: match[2],
                },
              };
            }
          }
          return {
            type: 'image',
            source: { type: 'url', url: url },
          };
        }
        return { type: 'text', text: '' };
      });
    } else {
      content = '';
    }

    if (msg.role === 'assistant' && msg.tool_calls) {
      const blocks = Array.isArray(content)
        ? content
        : [{ type: 'text', text: content }];

      for (const toolCall of msg.tool_calls) {
        blocks.push({
          type: 'tool_use',
          id: toolCall.id,
          name: toolCall.function.name,
          input: JSON.parse(toolCall.function.arguments),
        });
      }

      content = blocks;
    }

    messages.push({
      role: msg.role === 'assistant' ? 'assistant' : 'user',
      content,
    });
  }

  return { system, messages };
}

// Convert tools
function convertTools(
  openAITools?: Array<{
    type: string;
    function: { name: string; description: string; parameters: unknown };
  }>,
): any[] | undefined {
  if (!openAITools || openAITools.length === 0) return undefined;

  return openAITools.map((tool) => ({
    name: tool.function.name,
    description: tool.function.description,
    input_schema: tool.function.parameters,
  }));
}

// Map model names
function mapModelToBedrock(model: string): string {
  const modelName = model.includes('/') ? model.split('/').pop()! : model;

  if (modelName.includes('opus')) {
    return 'anthropic.claude-3-opus-20240229-v1:0';
  }
  if (modelName.includes('sonnet') || modelName.includes('4')) {
    return 'global.anthropic.claude-sonnet-4-5-20250929-v1:0';
  }
  if (modelName.includes('haiku')) {
    return 'anthropic.claude-3-haiku-20240307-v1:0';
  }

  return 'global.anthropic.claude-sonnet-4-5-20250929-v1:0';
}

// Non-streaming API call
export async function callBedrock(params: {
  model: string;
  messages: OpenAIMessage[];
  tools?: Array<{
    type: string;
    function: { name: string; description: string; parameters: unknown };
  }>;
  max_tokens?: number;
}): Promise<{
  choices: Array<{
    message: {
      role: string;
      content: string;
      tool_calls?: Array<{
        id: string;
        type: 'function';
        function: { name: string; arguments: string };
      }>;
    };
  }>;
}> {
  const { system, messages } = convertMessages(params.messages);
  const tools = convertTools(params.tools);
  const modelId = mapModelToBedrock(params.model);

  const body: any = {
    anthropic_version: 'bedrock-2023-05-31',
    max_tokens: params.max_tokens || 4096,
    messages,
  };

  if (system) body.system = system;
  if (tools) body.tools = tools;

  const command = new InvokeModelCommand({
    modelId,
    body: JSON.stringify(body),
    contentType: 'application/json',
    accept: 'application/json',
  });

  const response = await bedrockClient.send(command);
  const responseBody = JSON.parse(new TextDecoder().decode(response.body));

  // Convert to OpenAI format
  const content = responseBody.content || [];
  let textContent = '';
  const tool_calls: Array<{
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
  }> = [];

  for (const block of content) {
    if (block.type === 'text') {
      textContent += block.text;
    } else if (block.type === 'tool_use') {
      tool_calls.push({
        id: block.id,
        type: 'function',
        function: {
          name: block.name,
          arguments: JSON.stringify(block.input),
        },
      });
    }
  }

  const result: any = {
    message: {
      role: 'assistant',
      content: textContent,
    },
  };

  if (tool_calls.length > 0) {
    result.message.tool_calls = tool_calls;
  }

  return { choices: [result] };
}

// Streaming API call
export async function streamBedrock(params: {
  model: string;
  messages: OpenAIMessage[];
  tools?: Array<{
    type: string;
    function: { name: string; description: string; parameters: unknown };
  }>;
  max_tokens?: number;
}): Promise<ReadableStream> {
  const { system, messages } = convertMessages(params.messages);
  const tools = convertTools(params.tools);
  const modelId = mapModelToBedrock(params.model);

  const body: any = {
    anthropic_version: 'bedrock-2023-05-31',
    max_tokens: params.max_tokens || 4096,
    messages,
  };

  if (system) body.system = system;
  if (tools) body.tools = tools;

  const command = new InvokeModelWithResponseStreamCommand({
    modelId,
    body: JSON.stringify(body),
    contentType: 'application/json',
    accept: 'application/json',
  });

  const response = await bedrockClient.send(command);

  if (!response.body) {
    throw new Error('No response body from Bedrock');
  }

  // Transform Bedrock stream to OpenAI format
  return new ReadableStream({
    async start(controller) {
      let currentToolUse: { id: string; name: string; input: string } | null = null;

      try {
        for await (const event of response.body!) {
          if (event.chunk) {
            const chunk = JSON.parse(new TextDecoder().decode(event.chunk.bytes));

            if (chunk.type === 'content_block_start') {
              if (chunk.content_block?.type === 'tool_use') {
                currentToolUse = {
                  id: chunk.content_block.id,
                  name: chunk.content_block.name,
                  input: '',
                };
              }
            } else if (chunk.type === 'content_block_delta') {
              if (chunk.delta?.type === 'text_delta') {
                const openAIChunk = {
                  choices: [
                    {
                      delta: { content: chunk.delta.text },
                      index: 0,
                    },
                  ],
                };
                controller.enqueue(
                  new TextEncoder().encode('data: ' + JSON.stringify(openAIChunk) + '\n\n'),
                );
              } else if (chunk.delta?.type === 'input_json_delta' && currentToolUse) {
                currentToolUse.input += chunk.delta.partial_json;
              }
            } else if (chunk.type === 'content_block_stop' && currentToolUse) {
              const openAIChunk = {
                choices: [
                  {
                    delta: {
                      tool_calls: [
                        {
                          index: 0,
                          id: currentToolUse.id,
                          type: 'function',
                          function: {
                            name: currentToolUse.name,
                            arguments: currentToolUse.input,
                          },
                        },
                      ],
                    },
                    index: 0,
                  },
                ],
              };
              controller.enqueue(
                new TextEncoder().encode('data: ' + JSON.stringify(openAIChunk) + '\n\n'),
              );
              currentToolUse = null;
            }
          }
        }

        controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        controller.close();
      } catch (error) {
        console.error('Bedrock stream error:', error);
        controller.error(error);
      }
    },
  });
}
