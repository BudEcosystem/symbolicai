# Context-Aware Matching Guide

## Overview

Context-aware matching validates that expressions match not just the pattern, but also the surrounding semantic context. This guide provides comprehensive insights based on extensive testing across industries.

## 🎯 Quick Start Recommendations

### Default Settings
```python
from semantic_bud_expressions import ContextAwareExpression

# Recommended defaults for most use cases
expr = ContextAwareExpression(
    expression="I {action} {item}",
    expected_context="shopping e-commerce online store",
    context_threshold=0.5,        # Balanced threshold
    context_window='auto',        # Intelligent context extraction
    registry=registry
)
```

## 📊 Optimal Thresholds by Use Case

Based on extensive testing across industries:

### 1. **General Purpose** (Threshold: 0.45-0.50)
- Balanced precision and recall
- Works well across industries
- Good starting point for most applications

### 2. **High Precision Applications** (Threshold: 0.55-0.65)
**Use when false positives are costly:**
- Medical diagnosis systems
- Financial transactions
- Legal document processing
- Security/access control

**Example:**
```python
# Medical context - high precision needed
medical_expr = ContextAwareExpression(
    expression="patient has {condition}",
    expected_context="medical healthcare hospital diagnosis",
    context_threshold=0.6  # Higher threshold for safety
)
```

### 3. **High Recall Applications** (Threshold: 0.35-0.45)
**Use when missing matches is costly:**
- E-commerce search
- Content discovery
- Customer support routing
- Information retrieval

**Example:**
```python
# E-commerce - catch more potential matches
shopping_expr = ContextAwareExpression(
    expression="add {product} to cart",
    expected_context="shopping purchase buy online",
    context_threshold=0.4  # Lower threshold for coverage
)
```

## 🏭 Industry-Specific Findings

### Healthcare
- **Optimal threshold**: 0.5-0.6
- **Key contexts**: "medical", "healthcare", "hospital", "doctor", "patient"
- **Challenge**: Distinguishing medical vs veterinary contexts
- **Solution**: Include specific medical terms in expected context

### Finance
- **Optimal threshold**: 0.45-0.55
- **Key contexts**: "banking", "financial", "payment", "transaction", "money"
- **Challenge**: Distinguishing from other types of "transfers"
- **Solution**: Include financial-specific terms like "account", "funds"

### E-commerce
- **Optimal threshold**: 0.4-0.5
- **Key contexts**: "shopping", "cart", "purchase", "buy", "online", "store"
- **Challenge**: Common verbs like "add" used in many contexts
- **Solution**: Rich context with e-commerce specific terms

### Education
- **Optimal threshold**: 0.4-0.5
- **Key contexts**: "school", "education", "learning", "classroom", "teacher"
- **Challenge**: Different education levels (K-12 vs university)
- **Solution**: Include level-specific terms when needed

### Legal
- **Optimal threshold**: 0.5-0.6
- **Key contexts**: "legal", "contract", "agreement", "law", "court"
- **Challenge**: Formal language patterns appear in other domains
- **Solution**: Use domain-specific legal terminology

## 📏 Context Window Strategies

### 1. **Auto (Recommended)**
```python
context_window='auto'  # Intelligently determines context
```
- Best for most use cases
- Adapts to text structure
- Balances context quality and performance

### 2. **Sentence-based**
```python
context_window='sentence'  # Previous sentence
```
- Good for structured text
- Works well with formal documents
- May miss context in fragmented text

### 3. **Word Count**
```python
context_window=50  # Previous 50 words
```
- Consistent context size
- Good for short texts
- Performance scales with window size

### 4. **Custom Strategies**
```python
# For long documents
context_window=200  # More context for better accuracy

# For chat/dialogue
context_window='sentence'  # Each message is typically complete
```

## 🔍 Finding Your Optimal Threshold

### Step 1: Collect Test Data
```python
positive_examples = [
    # Examples that SHOULD match your context
    "In our online store, add iPhone to cart",
    "Complete purchase by adding item to cart"
]

negative_examples = [
    # Examples that should NOT match
    "Add sugar to the recipe",
    "Add user to database"
]
```

### Step 2: Test Different Thresholds
```python
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
results = {}

for threshold in thresholds:
    expr = ContextAwareExpression(
        expression=your_expression,
        expected_context=your_context,
        context_threshold=threshold
    )
    
    # Test accuracy
    correct = 0
    for text in positive_examples:
        if expr.match_with_context(text):
            correct += 1
    for text in negative_examples:
        if not expr.match_with_context(text):
            correct += 1
    
    accuracy = correct / (len(positive_examples) + len(negative_examples))
    results[threshold] = accuracy
```

### Step 3: Analyze Results
- Plot accuracy vs threshold
- Consider false positive/negative costs
- Test with edge cases

## 💡 Best Practices

### 1. **Context Design**
```python
# Good: Specific and descriptive
expected_context="medical healthcare hospital diagnosis treatment"

# Poor: Too generic
expected_context="health help"
```

### 2. **Expression Patterns**
```python
# Be specific with parameter names
"patient has {symptom}"  # Clear medical context
"{person} has {thing}"   # Too generic
```

### 3. **Performance Optimization**
```python
# Cache expressions for reuse
expressions_cache = {}

def get_expression(pattern, context):
    key = f"{pattern}:{context}"
    if key not in expressions_cache:
        expressions_cache[key] = ContextAwareExpression(
            expression=pattern,
            expected_context=context,
            registry=registry
        )
    return expressions_cache[key]
```

### 4. **Debugging Context Matches**
```python
# Enable detailed debugging
match = expr.match_with_context(text)
if match:
    print(f"Matched with similarity: {match.context_similarity}")
    print(f"Context extracted: {match.context_text}")
else:
    # Manually check why it didn't match
    test_expr = ContextAwareExpression(
        expression=expression,
        expected_context=expected_context,
        context_threshold=0.0  # Accept all
    )
    test_match = test_expr.match_with_context(text)
    if test_match:
        print(f"Would match at threshold: {test_match.context_similarity}")
```

## 📈 Performance Characteristics

### Latency by Text Length
- **Short (1-50 words)**: ~0.5-2ms
- **Medium (50-200 words)**: ~1-5ms
- **Long (200+ words)**: ~2-10ms

### Factors Affecting Performance
1. Context window size
2. Model initialization (one-time cost)
3. Expression complexity
4. Cache hit rate

## 🚨 Common Pitfalls and Solutions

### 1. **Context Extraction Failures**
**Problem**: No context extracted
**Solution**: Check text structure, ensure proper sentence boundaries

### 2. **Too Strict Thresholds**
**Problem**: Valid matches rejected
**Solution**: Lower threshold or enrich expected context

### 3. **Too Loose Thresholds**
**Problem**: False positives
**Solution**: Increase threshold or add negative context terms

### 4. **Ambiguous Contexts**
**Problem**: Similar terms across domains
**Solution**: Use domain-specific terminology

## 🔧 Advanced Techniques

### 1. **Negative Context Filtering**
```python
# Future feature idea
expr = ContextAwareExpression(
    expression="transfer {amount}",
    expected_context="banking financial",
    negative_context="medical sports",  # Reject these contexts
    context_threshold=0.5
)
```

### 2. **Multi-Level Context**
```python
# Combine multiple context checks
financial_context = ContextAwareExpression(...)
secure_context = ContextAwareExpression(...)

if financial_context.match_with_context(text) and secure_context.match_with_context(text):
    # High confidence match
    process_transaction()
```

### 3. **Dynamic Threshold Adjustment**
```python
# Adjust based on confidence requirements
def get_threshold(risk_level):
    thresholds = {
        'low': 0.4,
        'medium': 0.5,
        'high': 0.6,
        'critical': 0.7
    }
    return thresholds.get(risk_level, 0.5)
```

## 📚 Summary

1. **Start with threshold 0.5** and adjust based on results
2. **Use 'auto' context window** for most cases
3. **Test with your specific domain data**
4. **Consider false positive vs false negative costs**
5. **Monitor and adjust based on production feedback**

The context-aware matching system provides powerful semantic validation beyond simple pattern matching. With proper threshold tuning and context design, it can significantly improve the accuracy and reliability of your text processing applications.