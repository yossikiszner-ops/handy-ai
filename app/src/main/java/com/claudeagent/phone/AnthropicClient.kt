package com.claudeagent.phone

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

data class AnthropicResponse(
    val stopReason: String?,
    val content: JsonArray,
    val usage: JsonObject?,
)

/**
 * Gemini-backed transport that preserves Handy's existing Anthropic-shaped
 * internal message format. This lets AgentLoop and the execution/tooling layer
 * stay unchanged while the network provider is switched to Gemini.
 *
 * The current UI still validates keys with the legacy "sk-ant-" prefix. To
 * keep this patch surgical, users can enter: sk-ant-<GEMINI_API_KEY>.
 * This client strips that compatibility prefix before calling Google.
 *
 * Gemini 3 requires thoughtSignature metadata from functionCall parts to be
 * returned verbatim on the next request. We keep that opaque signature inside
 * Handy's internal block and restore it when converting history back to Gemini.
 */
class AnthropicClient(apiKey: String) {

    private val geminiApiKey = apiKey.removePrefix(COMPAT_KEY_PREFIX)

    private val http = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    suspend fun sendMessage(
        model: String,
        systemPrompt: String,
        tools: JsonArray,
        messages: JsonArray,
        maxTokens: Int = 2048,
    ): AnthropicResponse = withContext(Dispatchers.IO) {
        val toolNameById = collectToolNames(messages)
        val contents = convertMessages(messages, toolNameById)
        val functionDeclarations = buildJsonArray {
            tools.forEach { el ->
                val src = el.jsonObject
                add(buildJsonObject {
                    src["name"]?.let { put("name", it) }
                    src["description"]?.let { put("description", it) }
                    src["input_schema"]?.let { put("parameters", it) }
                })
            }
        }

        val body = buildJsonObject {
            put("systemInstruction", buildJsonObject {
                put("parts", buildJsonArray {
                    add(buildJsonObject { put("text", JsonPrimitive(systemPrompt)) })
                })
            })
            put("contents", contents)
            put("tools", buildJsonArray {
                add(buildJsonObject {
                    put("functionDeclarations", functionDeclarations)
                })
            })
            put("toolConfig", buildJsonObject {
                put("functionCallingConfig", buildJsonObject {
                    put("mode", JsonPrimitive("ANY"))
                })
            })
            put("generationConfig", buildJsonObject {
                put("maxOutputTokens", JsonPrimitive(maxTokens))
                put("temperature", JsonPrimitive(0.2))
            })
        }

        val request = Request.Builder()
            .url("https://generativelanguage.googleapis.com/v1beta/models/$GEMINI_MODEL:generateContent?key=$geminiApiKey")
            .addHeader("content-type", "application/json")
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .build()

        var lastError: Throwable? = null
        repeat(MAX_ATTEMPTS) { attempt ->
            try {
                awaitCall(http.newCall(request)).use { resp ->
                    val text = resp.body?.string().orEmpty()
                    if (resp.code in RETRYABLE_STATUS) {
                        lastError = RuntimeException("Gemini API ${resp.code}: ${text.take(200)}")
                        return@use
                    }
                    if (!resp.isSuccessful) {
                        throw RuntimeException("Gemini API error ${resp.code}: ${text.take(500)}")
                    }

                    val root = json.parseToJsonElement(text).jsonObject
                    val candidate = root["candidates"]?.jsonArray?.firstOrNull()?.jsonObject
                    val parts = candidate
                        ?.get("content")?.jsonObject
                        ?.get("parts")?.jsonArray
                        ?: JsonArray(emptyList())

                    val anthropicContent = buildJsonArray {
                        parts.forEachIndexed { index, partEl ->
                            val part = partEl.jsonObject
                            val thoughtSignature = part["thoughtSignature"]

                            part["text"]?.jsonPrimitive?.content?.takeIf { it.isNotBlank() }?.let { t ->
                                add(buildJsonObject {
                                    put("type", JsonPrimitive("text"))
                                    put("text", JsonPrimitive(t))
                                    thoughtSignature?.let { put(GEMINI_SIGNATURE_FIELD, it) }
                                })
                            }

                            part["functionCall"]?.jsonObject?.let { call ->
                                val name = call["name"]?.jsonPrimitive?.content.orEmpty()
                                val args = call["args"]?.jsonObject ?: buildJsonObject { }
                                val id = call["id"]?.jsonPrimitive?.content
                                    ?.takeIf { it.isNotBlank() }
                                    ?: "gemini_${System.nanoTime()}_$index"
                                add(buildJsonObject {
                                    put("type", JsonPrimitive("tool_use"))
                                    put("id", JsonPrimitive(id))
                                    put("name", JsonPrimitive(name))
                                    put("input", args)
                                    thoughtSignature?.let { put(GEMINI_SIGNATURE_FIELD, it) }
                                })
                            }
                        }
                    }

                    return@withContext AnthropicResponse(
                        stopReason = candidate?.get("finishReason")?.jsonPrimitive?.content,
                        content = anthropicContent,
                        usage = root["usageMetadata"]?.jsonObject,
                    )
                }
            } catch (io: IOException) {
                lastError = io
            }
            if (attempt < MAX_ATTEMPTS - 1) delay(BACKOFF_MS[attempt])
        }
        throw lastError ?: RuntimeException("Gemini request failed")
    }

    private fun collectToolNames(messages: JsonArray): Map<String, String> {
        val out = mutableMapOf<String, String>()
        messages.forEach { msgEl ->
            val content = msgEl.jsonObject["content"] as? JsonArray ?: return@forEach
            content.forEach { blockEl ->
                val block = blockEl.jsonObject
                if (block["type"]?.jsonPrimitive?.content == "tool_use") {
                    val id = block["id"]?.jsonPrimitive?.content.orEmpty()
                    val name = block["name"]?.jsonPrimitive?.content.orEmpty()
                    if (id.isNotBlank() && name.isNotBlank()) out[id] = name
                }
            }
        }
        return out
    }

    private fun convertMessages(
        messages: JsonArray,
        toolNameById: Map<String, String>,
    ): JsonArray = buildJsonArray {
        messages.forEach { msgEl ->
            val msg = msgEl.jsonObject
            val role = when (msg["role"]?.jsonPrimitive?.content) {
                "assistant" -> "model"
                else -> "user"
            }
            val sourceContent = msg["content"] as? JsonArray ?: JsonArray(emptyList())
            val parts = buildJsonArray {
                sourceContent.forEach { blockEl ->
                    val block = blockEl.jsonObject
                    when (block["type"]?.jsonPrimitive?.content) {
                        "text" -> {
                            val text = block["text"]?.jsonPrimitive?.content.orEmpty()
                            add(buildJsonObject {
                                put("text", JsonPrimitive(text))
                                block[GEMINI_SIGNATURE_FIELD]?.let { put("thoughtSignature", it) }
                            })
                        }

                        "image" -> {
                            anthropicImageToGeminiPart(block)?.let { add(it) }
                        }

                        "tool_use" -> {
                            val id = block["id"]?.jsonPrimitive?.content.orEmpty()
                            val name = block["name"]?.jsonPrimitive?.content.orEmpty()
                            val input = block["input"]?.jsonObject ?: buildJsonObject { }
                            add(buildJsonObject {
                                put("functionCall", buildJsonObject {
                                    if (id.isNotBlank()) put("id", JsonPrimitive(id))
                                    put("name", JsonPrimitive(name))
                                    put("args", input)
                                })
                                block[GEMINI_SIGNATURE_FIELD]?.let { put("thoughtSignature", it) }
                            })
                        }

                        "tool_result" -> {
                            val id = block["tool_use_id"]?.jsonPrimitive?.content.orEmpty()
                            val name = toolNameById[id] ?: "unknown_tool"
                            val resultContent = block["content"] as? JsonArray ?: JsonArray(emptyList())
                            val resultText = resultContent
                                .filter { it.jsonObject["type"]?.jsonPrimitive?.content == "text" }
                                .joinToString("\n") { it.jsonObject["text"]?.jsonPrimitive?.content.orEmpty() }
                            val isError = block["is_error"]?.jsonPrimitive?.content?.toBooleanStrictOrNull() ?: false

                            add(buildJsonObject {
                                put("functionResponse", buildJsonObject {
                                    if (id.isNotBlank()) put("id", JsonPrimitive(id))
                                    put("name", JsonPrimitive(name))
                                    put("response", buildJsonObject {
                                        put(if (isError) "error" else "result", JsonPrimitive(resultText))
                                    })
                                })
                            })

                            resultContent.forEach { childEl ->
                                val child = childEl.jsonObject
                                if (child["type"]?.jsonPrimitive?.content == "image") {
                                    anthropicImageToGeminiPart(child)?.let { add(it) }
                                }
                            }
                        }
                    }
                }
            }

            if (parts.isNotEmpty()) {
                add(buildJsonObject {
                    put("role", JsonPrimitive(role))
                    put("parts", parts)
                })
            }
        }
    }

    private fun anthropicImageToGeminiPart(block: JsonObject): JsonObject? {
        val source = block["source"]?.jsonObject ?: return null
        val mediaType = source["media_type"]?.jsonPrimitive?.content ?: "image/jpeg"
        val data = source["data"]?.jsonPrimitive?.content ?: return null
        return buildJsonObject {
            put("inlineData", buildJsonObject {
                put("mimeType", JsonPrimitive(mediaType))
                put("data", JsonPrimitive(data))
            })
        }
    }

    private suspend fun awaitCall(call: okhttp3.Call): okhttp3.Response =
        suspendCancellableCoroutine { cont ->
            cont.invokeOnCancellation { runCatching { call.cancel() } }
            call.enqueue(object : okhttp3.Callback {
                override fun onFailure(c: okhttp3.Call, e: IOException) {
                    if (cont.isActive) cont.resumeWithException(e)
                }

                override fun onResponse(c: okhttp3.Call, response: okhttp3.Response) {
                    if (cont.isActive) cont.resume(response) else response.close()
                }
            })
        }

    companion object {
        private const val GEMINI_MODEL = "gemini-3.1-flash-lite"
        private const val COMPAT_KEY_PREFIX = "sk-ant-"
        private const val GEMINI_SIGNATURE_FIELD = "_gemini_thought_signature"
        private const val MAX_ATTEMPTS = 4
        private val BACKOFF_MS = longArrayOf(1_500L, 4_000L, 10_000L)
        private val RETRYABLE_STATUS = setOf(408, 429, 500, 502, 503, 504)
    }
}
