import { createHash } from "node:crypto";
import { config, log } from "./config.ts";
import type { ContentBlock, InputMessage, SystemPrompt } from "./types.ts";

/**
 * Collapses the ways a client can spell the same turn (bare string vs. a single text block,
 * extra vendor fields on a block) so that an echoed-back reply still matches the session that
 * produced it.
 */
function canonical(messages: InputMessage[]): unknown[] {
  return messages.map((m) => {
    const blocks: ContentBlock[] =
      typeof m.content === "string" ? [{ type: "text", text: m.content }] : m.content;
    const parts = blocks.map((b) =>
      b.type === "text" ? { type: "text", text: (b as { text: string }).text } : b,
    );
    if (parts.length === 1 && parts[0]!.type === "text") {
      return { role: m.role, content: (parts[0] as { text: string }).text };
    }
    return { role: m.role, content: parts };
  });
}

interface Entry {
  sessionId: string;
  expiresAt: number;
  busy: boolean;
}

/**
 * The Messages API is stateless: every request carries the whole transcript. The CLI is
 * stateful and can `--resume`. This cache bridges the two by fingerprinting the
 * conversation prefix a CLI session has already seen, so a follow-up request replays only
 * the newest user turn instead of re-sending the entire history.
 */
export class SessionCache {
  private readonly entries = new Map<string, Entry>();

  static key(model: string, system: SystemPrompt | undefined, messages: InputMessage[]): string {
    const h = createHash("sha256");
    h.update(JSON.stringify({ model, system: system ?? null, messages: canonical(messages) }));
    return h.digest("hex");
  }

  /** Claims the session for a conversation prefix. Returns null on a miss or if it is in use. */
  acquire(key: string): string | null {
    const entry = this.entries.get(key);
    if (!entry) return null;
    if (entry.expiresAt < Date.now()) {
      this.entries.delete(key);
      return null;
    }
    // A CLI session is single-writer; a concurrent turn must start its own.
    if (entry.busy) return null;
    entry.busy = true;
    // Refresh LRU position.
    this.entries.delete(key);
    this.entries.set(key, entry);
    return entry.sessionId;
  }

  /** Releases a claimed session without advancing it (used when a turn fails). */
  release(key: string): void {
    const entry = this.entries.get(key);
    if (entry) entry.busy = false;
  }

  /** Records that `sessionId` has now also seen the turn identified by `nextKey`. */
  commit(prevKey: string | null, nextKey: string, sessionId: string): void {
    if (prevKey !== null) this.entries.delete(prevKey);
    this.entries.set(nextKey, {
      sessionId,
      expiresAt: Date.now() + config.sessionTtlMs,
      busy: false,
    });
    this.evict();
  }

  private evict(): void {
    const now = Date.now();
    for (const [k, v] of this.entries) {
      if (v.expiresAt < now && !v.busy) this.entries.delete(k);
    }
    while (this.entries.size > config.sessionCacheMax) {
      // Map preserves insertion order, so the first key is the least recently used.
      const oldest = this.entries.keys().next();
      if (oldest.done) break;
      this.entries.delete(oldest.value);
      log("debug", "session cache evicted an entry");
    }
  }

  get size(): number {
    return this.entries.size;
  }
}

export const sessions = new SessionCache();
