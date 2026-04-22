// Anthropic API client for direct API calls
// Converts between OpenAI-style messages (used internally) and Anthropic format

const ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages';
const ANTHROPIC_API_KEY = Deno.env.get('ANTHROPIC_API_KEY') ?? '';
const ANTHROPIC_VERSION = '2023-06-01';

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

interface AnthropicMessage {
  role: 'user' | 'assistant';
  content:
    | string
    | Array<{
        type: 'text' | 'image' | 'tool_use' | 'tool_result';
        text?: string;
        source?: {
          type: 'url' | 'base64';
          url?: string;
          media_type?: string;
          data?: string;
        };
        id?: string;
        name?: string;
        input?: unknown;
        tool_use_id?: string;
        content?: string | unknown;
      }>;
}

interface AnthropicTool {
  name: string;
  description: string;
  input_schema: {
    type: 'object';
    properties: Record<string, unknown>;
    required?: string[];
  };
}

// Convert OpenAI messages to Anthropic format
function convertMessages(
  openAIMessages: OpenAIMessage[],
): { system?: string; messages: AnthropicMessage[] } {
  let system: string | undefined;
  const messages: AnthropicMessage[] = [];

  for (const msg of openAIMessages) {
    // Extract system message
    if (msg.role === 'system') {
      system = typeof msg.content === 'string' ? msg.content : '';
      continue;
    }

    // Skip tool role messages in conversion (handled separately)
    if (msg.role === 'tool') {
      continue;
    }

    // Convert content
    let content: AnthropicMessage['content'];

    if (typeof msg.content === 'string') {
      content = msg.content;
    } else if (Array.isArray(msg.content)) {
      // Convert multimodal content
      content = msg.content.map((part) => {
        if (part.type === 'text') {
          return { type: 'text', text: part.text || '' };
        } else if (part.type === 'image_url' && part.image_url) {
          const url = part.image_url.url;
          if (url.startsWith('data:')) {
            // Base64 image
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
          // URL image
          return {
            type: 'image',
            source: {
              type: 'url',
              url: url,
            },
          };
        }
        return { type: 'text', text: '' };
      });
    } else {
      content = '';
    }

    // Handle tool calls in assistant messages
    if (msg.role === 'assistant' && msg.tool_calls) {
      const blocks = Array.isArray(content) ? content : [{ type: 'text', text: content }];

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

// Convert OpenAI tool format to Anthropic format
function convertTools(
  openAITools?: Array<{ type: string; function: { name: string; description: string; parameters: unknown } }>,
): AnthropicTool[] | undefined {
  if (!openAITools || openAITools.length === 0) return undefined;

  return openAITools.map((tool) => ({
    name: tool.function.name,
    description: tool.function.description,
    input_schema: tool.function.parameters as AnthropicTool['input_schema'],
  }));
}

// Non-streaming API call
export async function callAnthropic(params: {
  model: string;
  messages: OpenAIMessage[];
  tools?: Array<{ type: string; function: { name: string; description: string; parameters: unknown } }>;
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

  const body: Record<string, unknown> = {
    model: mapModelName(params.model),
    max_tokens: params.max_tokens || 4096,
    messages,
  };

  if (system) body.system = system;
  if (tools) body.tools = tools;

  const response = await fetch(ANTHROPIC_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': ANTHROPIC_API_KEY,
      'anthropic-version': ANTHROPIC_VERSION,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(`Anthropic API Error: ${response.status} - ${errorText}`);
    throw new Error(`Anthropic API error: ${response.statusText} (${response.status})`);
  }

  const data = await response.json();

  // Convert Anthropic response to OpenAI format
  const content = data.content || [];
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

  const result: {
    message: {
      role: string;
      content: string;
      tool_calls?: Array<{
        id: string;
        type: 'function';
        function: { name: string; arguments: string };
      }>;
    };
  } = {
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
export async function streamAnthropic(params: {
  model: string;
  messages: OpenAIMessage[];
  tools?: Array<{ type: string; function: { name: string; description: string; parameters: unknown } }>;
  max_tokens?: number;
}): Promise<ReadableStream> {
  const { system, messages } = convertMessages(params.messages);
  const tools = convertTools(params.tools);

  const body: Record<string, unknown> = {
    model: mapModelName(params.model),
    max_tokens: params.max_tokens || 4096,
    messages,
    stream: true,
  };

  if (system) body.system = system;
  if (tools) body.tools = tools;

  const response = await fetch(ANTHROPIC_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': ANTHROPIC_API_KEY,
      'anthropic-version': ANTHROPIC_VERSION,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(`Anthropic API Error: ${response.status} - ${errorText}`);
    throw new Error(`Anthropic API error: ${response.statusText} (${response.status})`);
  }

  if (!response.body) {
    throw new Error('No response body from Anthropic API');
  }

  // Transform Anthropic SSE stream to OpenAI-compatible format
  return new ReadableStream({
    async start(controller) {
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let currentToolUse: { id: string; name: string; input: string } | null = null;

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.trim() || line.startsWith(':')) continue;

            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') continue;

              try {
                const event = JSON.parse(data);

                // Handle different event types
                if (event.type === 'content_block_start') {
                  if (event.content_block?.type === 'tool_use') {
                    currentToolUse = {
                      id: event.content_block.id,
                      name: event.content_block.name,
                      input: '',
                    };
                  }
                } else if (event.type === 'content_block_delta') {
                  if (event.delta?.type === 'text_delta') {
                    // Text content - convert to OpenAI format
                    const chunk = {
                      choices: [
                        {
                          delta: {
                            content: event.delta.text,
                          },
                          index: 0,
                        },
                      ],
                    };
                    controller.enqueue(
                      new TextEncoder().encode('data: ' + JSON.stringify(chunk) + '\n\n'),
                    );
                  } else if (
                    event.delta?.type === 'input_json_delta' &&
                    currentToolUse
                  ) {
                    currentToolUse.input += event.delta.partial_json;
                  }
                } else if (event.type === 'content_block_stop' && currentToolUse) {
                  // Send tool call in OpenAI format
                  const chunk = {
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
                    new TextEncoder().encode('data: ' + JSON.stringify(chunk) + '\n\n'),
                  );
                  currentToolUse = null;
                }
              } catch (e) {
                console.error('Error parsing Anthropic stream event:', e);
              }
            }
          }
        }

        controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
        controller.close();
      } catch (error) {
        controller.error(error);
      }
    },
  });
}

// Map OpenRouter model names to Anthropic model names
function mapModelName(openRouterModel: string): string {
  // Extract the actual model name after the provider prefix
  if (openRouterModel.includes('/')) {
    const parts = openRouterModel.split('/');
    const modelName = parts[parts.length - 1];

    // Map common variations
    if (modelName.includes('opus')) return 'claude-opus-4-20250514';
    if (modelName.includes('sonnet')) return 'claude-sonnet-4-20250514';
    if (modelName.includes('haiku')) return 'claude-haiku-4-20250514';

    // Return as-is if it looks like a Claude model
    if (modelName.startsWith('claude-')) return modelName;
  }

  // Default fallback
  return 'claude-sonnet-4-20250514';
}
