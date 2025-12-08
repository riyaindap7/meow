// types.ts - Shared types for chat functionality

export interface Message {
  message_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  content_type?: "text" | "file" | "image";
  created_at: string;
  tokens_estimate?: number;
  attachments?: Array<{
    name: string;
    url: string;
    type: string;
  }>;
  sources?: Array<{
    text: string;
    source_file: string;
    page_idx: number;
    score: number;
    global_chunk_id?: string;
    document_id?: string;
    chunk_index?: number;
    section_hierarchy?: string;
    char_count?: number;
    word_count?: number;
  }>;
  meta?: Record<string, unknown>;
}

export interface Conversation {
  conversation_id: string;
  user_id?: string;
  title?: string;
  created_at: string;
  updated_at: string;
  archived: boolean;
  message_count?: number;
  messages?: Message[];
  metadata?: {
    locale?: string;
    device?: string;
    tags?: string[];
  };
  context?: {
    query?: string;
    retrieved_docs?: Array<{
      text: string;
      source_file: string;
      page_idx: number;
      score: number;
      global_chunk_id?: string;
      document_id?: string;
      chunk_index?: number;
      section_hierarchy?: string;
      char_count?: number;
      word_count?: number;
    }>;
  };
  settings?: {
    temperature?: number;
    top_k?: number;
  };
}

export interface ChatRequest {
  query: string;
  conversation_id?: string;
  top_k?: number;
  temperature?: number;
  include_history?: boolean;
}

export interface ChatResponse {
  query: string;
  answer: string;
  conversation_id: string;
  sources: Array<{
    text: string;
    source: string;
    page?: number;
    score?: number;
    document_id?: string;
  }>;
  search_latency_ms: number;
  llm_latency_ms: number;
  total_latency_ms: number;
  model_used?: string;
}

export interface ConversationsListResponse {
  conversations: Conversation[];
  total: number;
}

export interface ConversationMessagesResponse {
  conversation_id: string;
  messages: Message[];
}

export interface CreateConversationResponse {
  conversation_id: string;
  created_at: string;
  message_count: number;
}

export interface DeleteConversationResponse {
  status: "archived" | "deleted";
  conversation_id: string;
}
