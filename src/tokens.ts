import type { ContentBlock, InputMessage, SystemPrompt } from "./types.ts";

/**
 * Offline approximation of Anthropic's tokenizer (~3.8 chars/token for prose, with a
 * floor so short strings are not rounded to zero). The real tokenizer is not available
 * locally, so every count this module produces is an estimate.
 */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.max(1, Math.ceil(text.length / 3.8));
}

export function blockText(block: ContentBlock): string {
  switch (block.type) {
    case "text":
      return typeof (block as { text?: unknown }).text === "string" ? (block as { text: string }).text : "";
    case "thinking":
      return typeof (block as { thinking?: unknown }).thinking === "string"
        ? (block as { thinking: string }).thinking
        : "";
    case "tool_use":
      return JSON.stringify((block as { input?: unknown }).input ?? {});
    case "tool_result": {
      const content = (block as { content?: unknown }).content;
      if (typeof content === "string") return content;
      if (Array.isArray(content)) return content.map((b) => blockText(b as ContentBlock)).join("\n");
      return "";
    }
    case "image":
      // Images are billed by dimensions; assume a mid-size screenshot.
      return "";
    default:
      return "";
  }
}

export function contentToText(content: string | ContentBlock[]): string {
  if (typeof content === "string") return content;
  return content.map(blockText).filter(Boolean).join("\n");
}

function imageTokens(content: string | ContentBlock[]): number {
  if (typeof content === "string") return 0;
  return content.filter((b) => b.type === "image").length * 1600;
}

export function systemToText(system: SystemPrompt | undefined): string {
  if (!system) return "";
  if (typeof system === "string") return system;
  return system.map((b) => b.text ?? "").join("\n");
}

export function estimateRequestTokens(
  messages: InputMessage[],
  system: SystemPrompt | undefined,
): number {
  let total = estimateTokens(systemToText(system));
  for (const m of messages) {
    // Every turn carries a few tokens of role/delimiter overhead.
    total += 4 + estimateTokens(contentToText(m.content)) + imageTokens(m.content);
  }
  return total;
}
