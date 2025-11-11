# Benchmark Metrics Explained

## Metric Breakdown

### Client-Side Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **TTFT** | Time to First Token | First token arrival - Request start |
| **E2EL** | End-to-End Latency | Response complete - Request start |
| **TPOT** | Time Per Output Token | Average time between tokens |
| **ITL** | Inter-Token Latency | Time between consecutive tokens |

### Server-Side Metrics (from Groq API)

| Metric | Description | What It Reveals |
|--------|-------------|-----------------|
| **Queue Time** | Time waiting in server queue | Server load / capacity issues |
| **Prompt Time** | Time to process input prompt | Prompt complexity / length impact |
| **Completion Time** | Time to generate all tokens | Generation efficiency |
| **Server Total** | Total server processing time | Queue + Prompt + Completion |

### Calculated Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Network Overhead** | Network + streaming latency | Client E2EL - Server Total Time |

## Why Network Overhead Matters

**Network Overhead** = E2EL (client) - Server Total Time (server)

This reveals:
- Network latency (RTT to Groq servers)
- Streaming overhead (chunked transfer)
- Client-side processing delays

## Example Analysis

```
Language Comparison:
                    English         Japanese        Difference
TTFT (p50)         202.4 ms        237.9 ms        +35.5 ms  ← Japanese slower to start
E2EL (p50)         216.5 ms        254.0 ms        +37.5 ms  ← Overall slower
Server Total (p50)  84.5 ms         98.7 ms        +14.2 ms  ← Server takes longer for Japanese
Network (p50)      132.0 ms        130.6 ms        -1.4 ms   ← Similar network overhead
```

**Interpretation:**
- Japanese prompts take longer to process on server (+14ms at p50)
- Network overhead is similar (~130ms), so not the bottleneck
- The difference is mostly server-side completion time
- Both languages show consistent network behavior

## Language-Specific Insights

### Why Japanese May Be Slower

1. **Tokenization Overhead**
   - Japanese text often requires more tokens than English
   - CJK characters may be split into multiple tokens
   - More tokens = longer generation time

2. **Model Architecture**
   - LLMs may be primarily trained on English
   - Cross-lingual transfer can be less efficient
   - Reasoning/safety models may have English bias

3. **Output Length Differences**
   - Japanese responses may be longer in token count
   - Even if character count is similar
   - Check `completion_tokens` to verify

## Using Server-Side Metrics

### Queue Time Analysis
- **High queue time** → Server capacity issues
- **Variable queue time** → Bursty traffic patterns
- **Consistent queue time** → Stable server load

### Completion Time Analysis
- **Long completion time** → Complex generation / long outputs
- **Varies by language** → Model efficiency differences
- **Scales with output length** → Check token counts

### Network Overhead Analysis
- **High network time** → Geographic distance / poor connectivity
- **Consistent across languages** → Not language-specific
- **Spiky network time** → Packet loss / congestion

## Optimization Strategies

Based on metric analysis:

### If Queue Time Is High
- Reduce `max_concurrency`
- Implement rate limiting
- Use off-peak hours

### If Completion Time Is High
- Use smaller models (llama-3.1-8b vs 70b)
- Reduce max output length
- Use `reasoning_effort=low` for safety models

### If Network Overhead Is High
- Use a closer geographic region
- Optimize network path
- Consider batch processing
- Enable compression if available

### If Japanese Is Consistently Slower
- Profile token counts to confirm tokenization overhead
- Test with different models (some may be better for multilingual)
- Consider pre-processing for efficiency
- Use language-specific optimizations

## Multi-Language Benchmark Best Practices

1. **Use Same Prompt Count** - Ensure fair comparison
2. **Check Token Counts** - Normalize by tokens, not just requests
3. **Compare Server Metrics** - Isolate network from server differences
4. **Run Multiple Times** - Average across runs to reduce variance
5. **Match Categories** - Compare same jailbreak types across languages

## Reading the Output Tables

### TTFT Table
Shows how quickly the first token arrives - user's first perception of latency.

### E2EL Table
Shows total request time - what users actually experience end-to-end.

### Server-Side Breakdown
Shows where time is spent on server:
- **Queue Time** - Waiting for resources
- **Completion Time** - Actual generation
- **Server Total** - Everything server-side

### Network Overhead
Shows client-side and network latency - helps identify if bottleneck is server or network.

### Logging and Debugging

**Long Request Detection**: Requests taking >1 second are automatically logged with detailed breakdowns including:
- TTFT, E2EL, token counts
- Server-side timing breakdown
- Network overhead calculation
- Content preview (in debug mode)

**Logging Levels**:
- Default: WARNING (anomalies only)
- `--verbose`: INFO (per-request details)
- `--debug`: DEBUG (content previews)

**Timing Validation**: The system automatically validates:
- TTFT < E2EL consistency
- Server timing vs client timing anomalies
