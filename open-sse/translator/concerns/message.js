import { OPENAI_BLOCK } from "../schema/index.js";

// Text-only content works with providers that require string messages. Keep
// multimodal arrays intact so images and other content parts retain their order.
// Cache-aware targets retain blocks that carry cache_control.
export function collapseTextParts(parts, opts = {}) {
  if (opts.preserveCacheControl && parts.some(part => part?.cache_control)) return parts;
  return parts.length > 0 && parts.every(part => part?.type === OPENAI_BLOCK.TEXT)
    ? parts.map(part => part.text).join("\n")
    : parts;
}
