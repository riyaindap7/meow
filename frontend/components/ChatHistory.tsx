"use client";

import React, { useState, useEffect, useRef } from "react";
import { useTheme } from "@/lib/ThemeContext";

interface Message {
  message_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  sources?: Array<{
    doc_id: string;
    source: string;
    page: number;
    score: number;
    snippet: string;
  }>;
}

interface Conversation {
  conversation_id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  archived: boolean;
  message_count?: number;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function ChatHistory() {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  // State management
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load conversations on mount
  useEffect(() => {
    loadConversations();
  }, []);

  // Load messages when conversation changes
  useEffect(() => {
    if (selectedConversation) {
      loadMessages(selectedConversation);
    }
  }, [selectedConversation]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Fetch all conversations
  const loadConversations = async () => {
    try {
      const response = await fetch(`${API_BASE}/conversations`);
      if (!response.ok) throw new Error("Failed to load conversations");
      const data = await response.json();
      setConversations(data.conversations || []);
      
      // Auto-select first conversation
      if (data.conversations && data.conversations.length > 0) {
        setSelectedConversation(data.conversations[0].conversation_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error loading conversations");
    }
  };

  // Fetch messages for a conversation
  const loadMessages = async (conversationId: string) => {
    try {
      const response = await fetch(
        `${API_BASE}/conversations/${conversationId}/messages`
      );
      if (!response.ok) throw new Error("Failed to load messages");
      const data = await response.json();
      setMessages(data.messages || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error loading messages");
      setMessages([]);
    }
  };

  // Send new message
  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || loading) return;

    const query = inputValue;
    setInputValue("");
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          conversation_id: selectedConversation,
          top_k: 5,
          temperature: 0.1,
        }),
      });

      if (!response.ok) throw new Error("Failed to send message");
      const result = await response.json();

      // Update conversation ID if new conversation was created
      if (result.conversation_id !== selectedConversation) {
        setSelectedConversation(result.conversation_id);
      }

      // Reload messages
      await loadMessages(result.conversation_id || selectedConversation || "");
      await loadConversations();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error sending message");
    } finally {
      setLoading(false);
    }
  };

  // Create new conversation
  const createNewConversation = async () => {
    try {
      const response = await fetch(`${API_BASE}/conversations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "New Conversation" }),
      });

      if (!response.ok) throw new Error("Failed to create conversation");
      const data = await response.json();

      setSelectedConversation(data.conversation_id);
      setMessages([]);
      await loadConversations();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error creating conversation");
    }
  };

  // Delete conversation
  const deleteConversation = async (conversationId: string) => {
    try {
      const response = await fetch(
        `${API_BASE}/conversations/${conversationId}`,
        { method: "DELETE" }
      );

      if (!response.ok) throw new Error("Failed to delete conversation");

      if (selectedConversation === conversationId) {
        setSelectedConversation(null);
        setMessages([]);
      }

      await loadConversations();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error deleting conversation");
    }
  };

  const bgClass = isDark ? "bg-neutral-900" : "bg-white";
  const textClass = isDark ? "text-white" : "text-gray-900";
  const borderClass = isDark ? "border-neutral-700" : "border-gray-200";
  const hoverClass = isDark
    ? "hover:bg-neutral-800"
    : "hover:bg-gray-50";
  const messageUserBg = isDark ? "bg-cyan-600" : "bg-cyan-500";
  const messageAssistantBg = isDark ? "bg-neutral-800" : "bg-gray-100";
  const messageUserText = isDark ? "text-white" : "text-white";
  const messageAssistantText = isDark ? "text-gray-100" : "text-gray-900";

  return (
    <div className={`flex h-screen ${bgClass} ${textClass}`}>
      {/* Sidebar - Conversations List */}
      <div
        className={`w-64 border-r ${borderClass} flex flex-col ${isDark ? "bg-neutral-950" : "bg-gray-50"}`}
      >
        {/* New Chat Button */}
        <button
          onClick={createNewConversation}
          className={`m-4 px-4 py-2 rounded-lg font-semibold transition-colors ${
            isDark
              ? "bg-cyan-600 hover:bg-cyan-700 text-white"
              : "bg-cyan-500 hover:bg-cyan-600 text-white"
          }`}
        >
          + New Chat
        </button>

        {/* Conversations List */}
        <div className="flex-1 overflow-y-auto">
          {conversations.length === 0 ? (
            <div className={`p-4 text-center text-sm ${isDark ? "text-gray-400" : "text-gray-500"}`}>
              No conversations yet
            </div>
          ) : (
            conversations.map((conv) => (
              <div
                key={conv.conversation_id}
                className={`border-b ${borderClass}`}
              >
                <button
                  onClick={() => setSelectedConversation(conv.conversation_id)}
                  className={`w-full text-left p-3 transition-colors ${
                    selectedConversation === conv.conversation_id
                      ? isDark
                        ? "bg-neutral-800"
                        : "bg-gray-100"
                      : hoverClass
                  }`}
                >
                  <div className="text-sm font-medium truncate">{conv.title}</div>
                  <div
                    className={`text-xs mt-1 ${isDark ? "text-gray-400" : "text-gray-500"}`}
                  >
                    {new Date(conv.updated_at).toLocaleDateString()}
                  </div>
                </button>

                {/* Delete Button */}
                <button
                  onClick={() => deleteConversation(conv.conversation_id)}
                  className={`w-full text-left px-3 py-1 text-xs ${
                    isDark
                      ? "text-red-400 hover:text-red-300"
                      : "text-red-600 hover:text-red-700"
                  } transition-colors`}
                >
                  Delete
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className={`flex-1 flex flex-col`}>
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {error && (
            <div
              className={`p-4 rounded-lg ${
                isDark
                  ? "bg-red-900/30 text-red-300 border border-red-700"
                  : "bg-red-50 text-red-700 border border-red-200"
              }`}
            >
              {error}
            </div>
          )}

          {messages.length === 0 ? (
            <div
              className={`text-center py-12 ${isDark ? "text-gray-400" : "text-gray-500"}`}
            >
              <p className="text-lg font-medium">No messages yet</p>
              <p className="text-sm mt-2">Start a conversation to see chat history</p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div key={msg.message_id} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-2xl px-4 py-2 rounded-lg ${
                    msg.role === "user"
                      ? `${messageUserBg} ${messageUserText}`
                      : `${messageAssistantBg} ${messageAssistantText}`
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  
                  {/* Show sources if available */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2 pt-2 border-t border-current border-opacity-20">
                      <p className="text-xs font-semibold mb-1">Sources:</p>
                      {msg.sources.map((source, idx) => (
                        <div key={idx} className="text-xs opacity-80">
                          <span className="font-medium">{source.source}</span>
                          {source.page && ` (p. ${source.page})`}
                          {source.score && ` - Score: ${(source.score * 100).toFixed(0)}%`}
                        </div>
                      ))}
                    </div>
                  )}
                  
                  <p className={`text-xs mt-1 opacity-70`}>
                    {new Date(msg.created_at).toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className={`border-t ${borderClass} p-4`}>
          <form onSubmit={sendMessage} className="flex gap-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask a question..."
              disabled={loading}
              className={`flex-1 px-4 py-2 rounded-lg border ${borderClass} focus:outline-none focus:ring-2 focus:ring-cyan-500 ${
                isDark
                  ? "bg-neutral-800 text-white placeholder-gray-500"
                  : "bg-gray-50 text-gray-900 placeholder-gray-400"
              } disabled:opacity-50`}
            />
            <button
              type="submit"
              disabled={loading || !inputValue.trim()}
              className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                isDark
                  ? "bg-cyan-600 hover:bg-cyan-700 text-white disabled:opacity-50"
                  : "bg-cyan-500 hover:bg-cyan-600 text-white disabled:opacity-50"
              }`}
            >
              {loading ? "Sending..." : "Send"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
