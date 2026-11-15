/**
 * Web-cookie chat executors (Qwen web + DeepSeek token extraction).
 * Source: open-sse/executors/qwen-web.js, open-sse/executors/deepseek-web.js
 *
 * The DeepSeek web executor itself is upstream's (9router) implementation; its
 * prompt/PoW/stream behaviour is not re-tested here.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import { extractUserToken } from "../../open-sse/executors/deepseek-web.js";
import {
  mapQwenModel,
  isWafResponse,
  QwenWebExecutor,
  parseOpenAIMessages,
} from "../../open-sse/executors/qwen-web.js";

const originalFetch = global.fetch;

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

async function readSseText(response) {
  return await response.text();
}

describe("Qwen parseOpenAIMessages", () => {
  it("formats system+user+assistant+user with last user unprefixed", () => {
    const out = parseOpenAIMessages([
      { role: "system", content: "sys" },
      { role: "user", content: "u1" },
      { role: "assistant", content: "a1" },
      { role: "user", content: "u2" },
    ]);
    expect(out).toBe("system: sys\n\nuser: u1\n\nassistant: a1\n\nu2");
  });

  it("treats developer as system", () => {
    const out = parseOpenAIMessages([
      { role: "developer", content: "dev rules" },
      { role: "user", content: "hi" },
    ]);
    expect(out).toBe("system: dev rules\n\nhi");
  });

  it("joins array text parts", () => {
    const out = parseOpenAIMessages([
      { role: "user", content: [{ type: "text", text: "part1" }, { type: "text", text: "part2" }] },
    ]);
    expect(out).toBe("part1 part2");
  });
});

describe("extractUserToken", () => {
  it("unwraps JSON value form", () => {
    expect(extractUserToken({ apiKey: '{"value":"jwt-here"}' })).toBe("jwt-here");
  });
});

// ── Qwen ────────────────────────────────────────────────────────────────────

const QWEN_FULL_COOKIE =
  "cna=abc; ssxmod_itna=xyz; ssxmod_itna2=def; token=qwen-bearer-tok";

function qwenSseResponse(events) {
  const text =
    events.map((e) => `data: ${typeof e === "string" ? e : JSON.stringify(e)}\n\n`).join("") +
    "data: [DONE]\n\n";
  return new Response(text, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

/** Two-call fetch: chats/new → chat/completions */
function mockQwenFetchSequence({
  newChatStatus = 200,
  newChatBody = { data: { id: "c1" } },
  newChatContentType = "application/json",
  newChatText = null,
  completionEvents = [
    { choices: [{ delta: { phase: "think", content: "r" } }] },
    { choices: [{ delta: { phase: "answer", content: "hi" } }] },
  ],
  completionStatus = 200,
  completionThrow = false,
} = {}) {
  const calls = [];
  global.fetch = vi.fn(async (url, init) => {
    const u = String(url);
    calls.push({ url: u, method: init?.method || "GET", body: init?.body, headers: init?.headers });
    if (u.includes("/chats/new")) {
      if (newChatText != null) {
        return new Response(newChatText, {
          status: newChatStatus,
          headers: { "Content-Type": newChatContentType },
        });
      }
      if (newChatStatus !== 200) {
        return new Response(JSON.stringify(newChatBody), {
          status: newChatStatus,
          headers: { "Content-Type": newChatContentType },
        });
      }
      return jsonResponse(newChatBody, newChatStatus);
    }
    if (u.includes("/chat/completions")) {
      if (completionThrow) throw new Error("network down");
      if (completionStatus !== 200) {
        return jsonResponse({ error: "fail" }, completionStatus);
      }
      return qwenSseResponse(completionEvents);
    }
    return jsonResponse({ error: `unexpected url ${u}` }, 500);
  });
  return calls;
}

describe("mapQwenModel", () => {
  it('mapQwenModel("qwen-max") === "qwen3.7-max"', () => {
    expect(mapQwenModel("qwen-max")).toBe("qwen3.7-max");
  });

  it("aliases plus/turbo and default fallback", () => {
    expect(mapQwenModel("qwen-plus")).toBe("qwen3.7-plus");
    expect(mapQwenModel("qwen3-flash")).toBe("qwen3.6-plus");
    expect(mapQwenModel("qwen3.8-max-preview")).toBe("qwen3.8-max-preview");
  });
});

describe("isWafResponse", () => {
  it("HTML content-type → true", () => {
    expect(isWafResponse(200, "text/html", "<html>")).toBe(true);
  });

  it("504 → true", () => {
    expect(isWafResponse(504, "application/json", "{}")).toBe(true);
  });

  it("normal JSON 200 → false", () => {
    expect(isWafResponse(200, "application/json", "{}")).toBe(false);
  });
});

describe("QwenWebExecutor.execute", () => {
  let executor;
  beforeEach(() => {
    executor = new QwenWebExecutor();
  });
  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("happy stream: reasoning_content then content, ends with [DONE]", async () => {
    const calls = mockQwenFetchSequence();
    const { response } = await executor.execute({
      model: "qwen-max",
      body: { messages: [{ role: "user", content: "hello" }] },
      stream: true,
      credentials: { apiKey: QWEN_FULL_COOKIE },
    });
    expect(response.status).toBe(200);
    const text = await readSseText(response);
    expect(text).toContain("reasoning_content");
    expect(text).toMatch(/"reasoning_content"\s*:\s*"r"/);
    expect(text).toMatch(/"content"\s*:\s*"hi"/);
    expect(text).toContain("data: [DONE]");

    expect(calls.map((c) => c.url)).toEqual([
      expect.stringContaining("/chats/new"),
      expect.stringMatching(/\/chat\/completions\?chat_id=c1/),
    ]);
    expect(calls).toHaveLength(2);

    const newChatBody = JSON.parse(calls[0].body);
    expect(newChatBody).toMatchObject({
      title: "New Chat",
      models: ["qwen3.7-max"],
      chat_mode: "normal",
      chat_type: "t2t",
    });
    expect(typeof newChatBody.timestamp).toBe("number");

    const headers = calls[0].headers;
    expect(headers.source || headers.Source).toBe("web");
    expect(headers.version || headers.Version).toBe("0.2.66");
    expect(headers["bx-v"] || headers["Bx-V"]).toBe("2.5.36");
    expect(headers.Cookie || headers.cookie).toContain("token=qwen-bearer-tok");
    expect(headers.Authorization || headers.authorization).toMatch(/Bearer\s+qwen-bearer-tok/);
  });

  it("HTML on chats/new → 401 WAF, message has no raw HTML", async () => {
    mockQwenFetchSequence({
      newChatStatus: 200,
      newChatContentType: "text/html",
      newChatText: "<html><body>aliyun_waf block</body></html>",
    });
    const { response } = await executor.execute({
      model: "qwen-max",
      body: { messages: [{ role: "user", content: "hi" }] },
      stream: true,
      credentials: { apiKey: QWEN_FULL_COOKIE },
    });
    expect(response.status).toBe(401);
    const body = await response.json();
    const msg = body.error?.message || JSON.stringify(body);
    expect(msg).toMatch(/WAF|cookie/i);
    expect(msg).not.toMatch(/<html/i);
  });

  it("bare token credentials → 400, fetch not called", async () => {
    global.fetch = vi.fn();
    const { response } = await executor.execute({
      model: "qwen-max",
      body: { messages: [{ role: "user", content: "hi" }] },
      stream: true,
      credentials: { apiKey: "bare-token-only" },
    });
    expect(response.status).toBe(400);
    expect(global.fetch).not.toHaveBeenCalled();
    const body = await response.json();
    expect(body.error?.message || "").toMatch(/Cookie/i);
  });

  it("tools present → still text completion, two fetches, no tool_calls", async () => {
    const calls = mockQwenFetchSequence({
      completionEvents: [{ choices: [{ delta: { phase: "answer", content: "plain" } }] }],
    });
    const { response } = await executor.execute({
      model: "qwen-max",
      body: {
        messages: [{ role: "user", content: "hi" }],
        tools: [{ type: "function", function: { name: "Shell", description: "run" } }],
      },
      stream: true,
      credentials: { apiKey: QWEN_FULL_COOKIE },
    });
    const text = await readSseText(response);
    expect(text).toContain("plain");
    expect(text).not.toContain("tool_calls");
    expect(calls).toHaveLength(2);
  });

  it("missing data.id on chats/new → 502", async () => {
    mockQwenFetchSequence({ newChatBody: { data: {} } });
    const { response } = await executor.execute({
      model: "qwen-max",
      body: { messages: [{ role: "user", content: "hi" }] },
      stream: true,
      credentials: { apiKey: QWEN_FULL_COOKIE },
    });
    expect(response.status).toBe(502);
  });

  it("completion 429 → 429", async () => {
    mockQwenFetchSequence({ completionStatus: 429 });
    const { response } = await executor.execute({
      model: "qwen-max",
      body: { messages: [{ role: "user", content: "hi" }] },
      stream: true,
      credentials: { apiKey: QWEN_FULL_COOKIE },
    });
    expect(response.status).toBe(429);
  });

  it("completion fetch throw → 502", async () => {
    mockQwenFetchSequence({ completionThrow: true });
    const { response } = await executor.execute({
      model: "qwen-max",
      body: { messages: [{ role: "user", content: "hi" }] },
      stream: true,
      credentials: { apiKey: QWEN_FULL_COOKIE },
    });
    expect(response.status).toBe(502);
  });
});
