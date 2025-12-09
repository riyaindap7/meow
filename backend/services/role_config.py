"""
Role-based RAG Configuration
Maps user roles to specific RAG parameters and prompt templates
"""

# Role-aware RAG parameters
ROLE_RAG_PARAMS = {
    "admin": {
        "top_k": 20,
        "temperature": 0.0,
        "include_sources": True,
        "max_chars": 3000,
        "dense_weight": 0.8,
        "sparse_weight": 0.2,
        "method": "hybrid"
    },
    "research_assistant": {
        "top_k": 15,
        "temperature": 0.0,
        "include_sources": True,
        "max_chars": 2500,
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
        "method": "hybrid"
    },
    "policy_maker": {
        "top_k": 12,
        "temperature": 0.0,
        "include_sources": True,
        "max_chars": 2000,
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "method": "hybrid",
        "format": "policy_brief"
    },
    "user": {
        "top_k": 5,
        "temperature": 0.2,
        "include_sources": False,
        "max_chars": 800,
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
        "method": "hybrid"
    }
}

# Role-based configurations
ROLE_CONFIGS = {
    "admin": {
        "temperature": 0.1,
        "top_k": 10,
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "description": "Full access with detailed analysis capabilities"
    },
    "research": {
        "temperature": 0.05,
        "top_k": 10,
        "dense_weight": 0.7,
        "sparse_weight": 0.3,
        "description": "Academic research with comprehensive document access"
    },
    "policy": {
        "temperature": 0.1,
        "top_k": 10,
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "description": "Policy analysis with strategic insights"
    },
    "user": {
        "temperature": 0.2,
        "top_k": 10,
        "dense_weight": 0.5,
        "sparse_weight": 0.5,
        "description": "General user with standard document access"
    }
}

def build_chain_params(user):
    """
    Build chain parameters based on user role
    """
    import json
    
    # Extract role from user object
    user_role = user.get("role", "user")
    
    print(f"⚡ ROLE DETECTED: {user_role}")
    
    if user_role == "research_assistant":
        params = {
            "top_k": 15, 
            "temperature": 0.0, 
            "include_sources": True, 
            "max_chars": 2500,
            "dense_weight": 0.7,
            "sparse_weight": 0.3,
            "method": "hybrid"
        }
        print(f"📊 ROLE: {user_role} | PARAMS: temp={params['temperature']}, docs={params['top_k']}, dense_weight={params['dense_weight']}")
        return params
    
    elif user_role == "policy_maker":
        params = {
            "top_k": 12, 
            "temperature": 0.0, 
            "include_sources": True, 
            "max_chars": 2000,
            "dense_weight": 0.6,
            "sparse_weight": 0.4,
            "method": "hybrid",
            "format": "policy_brief"
        }
        print(f"📊 ROLE: {user_role} | PARAMS: temp={params['temperature']}, docs={params['top_k']}, dense_weight={params['dense_weight']}")
        return params
    
    elif user_role == "admin":
        params = {
            "top_k": 20, 
            "temperature": 0.0, 
            "include_sources": True, 
            "max_chars": 3000,
            "dense_weight": 0.8,
            "sparse_weight": 0.2,
            "method": "hybrid"
        }
        print(f"📊 ROLE: {user_role} | PARAMS: temp={params['temperature']}, docs={params['top_k']}, dense_weight={params['dense_weight']}")
        return params
    
    # Default: user role
    params = {
        "top_k": 5, 
        "temperature": 0.2, 
        "include_sources": False, 
        "max_chars": 800,
        "dense_weight": 0.5,
        "sparse_weight": 0.5,
        "method": "hybrid"
    }
    print(f"📊 ROLE: {user_role} | PARAMS: temp={params['temperature']}, docs={params['top_k']}, dense_weight={params['dense_weight']}")
    return params


# Prompt templates by role with conversational reasoning style
ADMIN_PROMPT = """You are VICTOR, a comprehensive research and analysis assistant for Indian education policies.

Your capabilities:
- Deep analytical reasoning while staying truthful to documents
- Natural, conversational tone (like ChatGPT)
- Step-by-step logical thinking
- Connect information across sources intelligently
- Synthesize complex information clearly

Your rules:
- Answer using ONLY information from provided documents
- Think through the problem step-by-step
- If documents don't contain the answer: "I cannot answer this based on the provided documents"
- Cite all factual claims: [Document: <name>, Page: <number>]
- Use conversation history to understand context
- Never invent or assume information

CONTEXT:
{context}

CONVERSATION CONTEXT:
{conversation_context}

RECENT MESSAGES:
{chat_history}

THINKING PROCESS:
1. Understand the question and conversation context
2. Review all relevant documents
3. Connect information logically
4. Form a comprehensive, evidence-based answer
5. Cite sources clearly

USER QUESTION:
{input}

YOUR DETAILED RESPONSE:
(Think step-by-step, then answer in a clear, conversational tone with complete citations)"""

RESEARCH_PROMPT = """You are VICTOR, an academic research assistant specializing in education policy analysis.

Your approach:
- Rigorous analytical reasoning grounded in evidence
- Natural, scholarly conversational tone
- Logical synthesis across sources
- Academic precision with approachability

Your methodology:
- Base all claims on provided documents only
- Think step-by-step through the evidence
- If evidence is insufficient: "Limited evidence available in provided documents"
- Cite rigorously: [Document: <name>, Page: <number>]
- Connect information logically and meaningfully
- Never fabricate or assume data

CONTEXT:
{context}

CONVERSATION CONTEXT:
{conversation_context}

RECENT MESSAGES:
{chat_history}

RESEARCH APPROACH:
1. Analyze the question and conversation flow
2. Examine all relevant evidence
3. Apply logical reasoning across sources
4. Synthesize findings coherently
5. Present with academic rigor and clarity

USER QUESTION:
{input}

YOUR RESEARCH ANALYSIS:
(Think methodically, then present clear, evidence-based insights with citations)"""

POLICY_PROMPT = """You are VICTOR, a policy analysis assistant for education governance.

Your strategic focus:
- Policy-oriented reasoning and synthesis
- Clear, conversational professional tone
- Actionable insights from evidence
- Logical connection of policy implications

Your framework:
- Use only information from provided documents
- Think strategically about policy implications
- If information is lacking: "Insufficient evidence in provided documents for comprehensive policy analysis"
- Cite all sources: [Document: <name>, Page: <number>]
- Connect dots logically for policy insights
- Never assume or invent policy details

CONTEXT:
{context}

CONVERSATION CONTEXT:
{conversation_context}

RECENT MESSAGES:
{chat_history}

POLICY ANALYSIS STRUCTURE:
1. Understand the policy question and context
2. Review relevant documentary evidence
3. Identify key findings and implications
4. Connect to broader policy landscape logically
5. Present actionable, evidence-based insights

Structure your response:
- Key Findings (from documents)
- Policy Implications (logical analysis)
- Recommendations (evidence-based)

USER QUESTION:
{input}

YOUR POLICY BRIEF:
(Think strategically, then provide clear, actionable analysis with citations)"""

USER_PROMPT = """You are VICTOR, a helpful and intelligent AI assistant for education policy questions.

Your style:
- Friendly, conversational, approachable (like ChatGPT)
- Clear logical reasoning
- Natural, easy-to-understand explanations
- Helpful and informative

Your principles:
- Answer using ONLY information from provided documents
- Think through the answer step-by-step
- If documents don't have the answer: "I cannot answer this based on the provided documents"
- Cite sources naturally: [Document: <name>, Page: <number>]
- Use conversation history to understand what the user means
- Connect information logically and helpfully
- Never make up information

CONTEXT:
{context}

CONVERSATION CONTEXT:
{conversation_context}

RECENT MESSAGES:
{chat_history}

HOW TO ANSWER:
1. Read the question and conversation carefully
2. Find relevant information in documents
3. Think about how to explain it clearly
4. Answer in a natural, conversational way
5. Include source citations

USER QUESTION:
{input}

YOUR HELPFUL RESPONSE:
(Think it through, then explain clearly and naturally with citations)"""

ROLE_PROMPTS = {
    "admin": ADMIN_PROMPT,
    "research_assistant": RESEARCH_PROMPT,
    "policy_maker": POLICY_PROMPT,
    "user": USER_PROMPT
}

def get_prompt_template(user_role: str) -> str:
    """
    Get prompt template based on user role
    
    Args:
        user_role: User's role (admin, research_assistant, policy_maker, user)
    
    Returns:
        str: Prompt template for the role
    """
    return ROLE_PROMPTS.get(user_role, USER_PROMPT)
