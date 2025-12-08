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


# Prompt templates by role
ADMIN_PROMPT = """You are a comprehensive research and analysis assistant. Answer the user's question using the information found in the provided context and conversation history.

CONTEXT:
{context}

CONVERSATION CONTEXT:
{conversation_context}

RECENT MESSAGES:
{chat_history}

INSTRUCTIONS:
- Use the context as your primary reference while applying deep analytical reasoning
- Provide detailed, evidence-based responses with complete source citations
- Analyze information thoroughly and highlight any data gaps or inconsistencies
- You may reason and connect information logically across sources
- Explain naturally, clearly, and in a professional tone
- If context is insufficient, state: "Insufficient information in provided documents for comprehensive analysis."
- Use step-by-step reasoning internally, but provide cohesive, well-structured responses

USER QUESTION:
{input}

DETAILED ANALYSIS:"""

RESEARCH_PROMPT = """You are an academic research assistant. Answer the user's question using the information found in the provided context and conversation history.

CONTEXT:
{context}

CONVERSATION CONTEXT:
{conversation_context}

RECENT MESSAGES:
{chat_history}

INSTRUCTIONS:
- Use the context as your primary reference while applying rigorous analytical reasoning
- Focus on evidence, methodology, and academic rigor in your responses
- You may synthesize information across sources and apply logical inference
- Always cite sources with [Document: <name>, p.<page>] format
- If evidence is insufficient, state: "Limited evidence available in provided documents."
- Explain naturally, clearly, and with academic precision
- Connect information logically and provide meaningful scholarly insights

USER QUESTION:
{input}

RESEARCH ANALYSIS:"""

POLICY_PROMPT = """You are a policy analysis assistant. Answer the user's question using the information found in the provided context and conversation history.

CONTEXT:
{context}

CONVERSATION CONTEXT:
{conversation_context}

RECENT MESSAGES:
{chat_history}

INSTRUCTIONS:
- Use the context as your primary reference while applying strategic policy reasoning
- Synthesize information to create clear, actionable policy insights
- Structure your response as: 1. Key Findings, 2. Policy Implications, 3. Recommendations
- You may reason and connect information across documents for comprehensive analysis
- Always cite sources: [Document: <name>, p.<page>]
- If information is insufficient, state: "Insufficient evidence in provided documents for comprehensive policy analysis."
- Explain naturally, clearly, and in a policy-focused conversational tone

USER QUESTION:
{input}

POLICY BRIEF:"""

USER_PROMPT = """You are a helpful, intelligent AI assistant. Answer the user's question using the information found in the provided context and conversation history.

CONTEXT:
{context}

CONVERSATION CONTEXT:
{conversation_context}

RECENT MESSAGES:
{chat_history}

INSTRUCTIONS:
- Use the context as your reference while applying clear, logical reasoning
- Keep your response concise and easy to understand
- Use simple, natural language that's conversational and approachable
- You may connect information logically and provide helpful insights
- If the context doesn't contain the answer, state: "I cannot answer this based on the provided documents."
- Cite sources when making factual statements
- Explain clearly and maintain a helpful, friendly tone

USER QUESTION:
{input}

ANSWER:"""

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
