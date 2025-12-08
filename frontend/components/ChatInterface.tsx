'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';

interface Message {
  message_id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
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
}

interface Conversation {
  conversation_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count?: number;
}

interface ChatResponse {
  answer: string;
  conversation_id: string;
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
}

interface ChatInterfaceProps {
  authToken: string;
}

export default function ChatInterface({ authToken }: ChatInterfaceProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversation, setCurrentConversation] = useState<string | null>(null);
  const [currentTitle, setCurrentTitle] = useState<string>('New Chat');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [topK, setTopK] = useState(3);
  const [temperature, setTemperature] = useState(0.1);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load conversation messages when selected
  useEffect(() => {
    if (currentConversation) {
      fetchMessages(currentConversation);
    }
  }, [currentConversation]);

  const loadConversations = useCallback(async () => {
    if (!authToken) return;
    
    try {
      const response = await fetch(`${API_URL}/conversations`, {
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        }
      });
      if (response.ok) {
        const data = await response.json();
        const convs = data.conversations || [];
        setConversations(convs);
        // Auto-select first conversation
        if (convs.length > 0 && !currentConversation) {
          setCurrentConversation(convs[0].conversation_id);
          setCurrentTitle(convs[0].title);
        }
      }
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  }, [authToken, API_URL, currentConversation]);

  // Load conversations on mount (only if authenticated)
  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  const fetchMessages = async (conversationId: string) => {
    try {
      console.log(`📥 Fetching messages for conversation: ${conversationId}`);
      const response = await fetch(`${API_URL}/conversations/${conversationId}/messages`, {
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        }
      });
      if (response.ok) {
        const data = await response.json();
        console.log(`✅ Loaded ${data.messages?.length || 0} messages`);
        console.log('📋 Messages data:', data.messages);
        setMessages(data.messages || []);
      } else {
        const errorData = await response.json();
        console.error('❌ Failed to load messages:', errorData);
      }
    } catch (error) {
      console.error('❌ Error loading messages:', error);
    }
  };

  const createNewChat = async () => {
    try {
      console.log('🆕 Creating new conversation...');
      const response = await fetch(`${API_URL}/conversations`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ title: 'New Conversation' })
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Conversation created:', data);
        
        const newConversation: Conversation = {
          conversation_id: data.conversation_id,
          title: data.title || 'New Conversation',
          created_at: data.created_at || new Date().toISOString(),
          updated_at: data.updated_at || new Date().toISOString(),
          message_count: data.message_count || 0
        };
        
        setConversations([newConversation, ...conversations]);
        setCurrentConversation(data.conversation_id);
        setCurrentTitle(data.title || 'New Conversation');
        setMessages([]);
        setInputValue('');
        console.log('✅ UI updated with new conversation');
      } else {
        const errorData = await response.json();
        console.error('❌ Failed to create conversation:', errorData);
        alert(`Failed to create new chat: ${errorData.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('❌ Error creating conversation:', error);
      alert('Failed to create new chat');
    }
  };

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || !currentConversation) {
      console.log('❌ Cannot send: No input or no conversation selected');
      return;
    }

    const userMessage = inputValue;
    setInputValue('');
    setLoading(true);

    try {
      console.log(`📤 Sending message to conversation: ${currentConversation}`);
      console.log(`   Query: "${userMessage}"`);
      
      // Add user message to UI immediately
      const userMsg: Message = {
        message_id: Date.now().toString(),
        role: 'user',
        content: userMessage,
        created_at: new Date().toISOString()
      };
      setMessages(prev => [...prev, userMsg]);

      // Send to API
      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: userMessage,
          conversation_id: currentConversation,
          top_k: topK,
          temperature
        })
      });

      if (response.ok) {
        const data: ChatResponse = await response.json();
        console.log(`✅ Received response from backend`);
        console.log(`   Full response data:`, data);
        console.log(`   Answer: "${data.answer}"`);
        console.log(`   Answer length: ${data.answer?.length || 0} chars`);
        console.log(`   Sources: ${data.sources?.length || 0}`);
        
        if (!data.answer) {
          console.error('⚠️ WARNING: No answer in response!');
          alert('Received empty response from backend');
          return;
        }
        
        // Add assistant message
        const assistantMsg: Message = {
          message_id: Date.now().toString() + '1',
          role: 'assistant',
          content: data.answer,
          created_at: new Date().toISOString(),
          sources: data.sources
        };
        console.log('📝 Adding assistant message to UI:', assistantMsg);
        setMessages(prev => {
          const newMessages = [...prev, assistantMsg];
          console.log(`   Total messages after add: ${newMessages.length}`);
          return newMessages;
        });

        // Update conversation title if it's still "New Conversation"
        if (currentTitle === 'New Conversation') {
          setCurrentTitle(userMessage.substring(0, 50));
          // Update in list
          setConversations(prev => 
            prev.map(conv => 
              conv.conversation_id === currentConversation 
                ? { ...conv, title: userMessage.substring(0, 50) }
                : conv
            )
          );
        }
      } else {
        const errorData = await response.json();
        console.error('❌ Backend returned error:', errorData);
        alert(`Failed to get response: ${errorData.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('❌ Error sending message:', error);
      alert('Error sending message');
    } finally {
      setLoading(false);
    }
  };

  const deleteConversation = async (conversationId: string) => {
    if (!window.confirm('Delete this conversation?')) return;

    try {
      await fetch(`${API_URL}/conversations/${conversationId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        }
      });
      
      setConversations(conversations.filter(c => c.conversation_id !== conversationId));
      if (currentConversation === conversationId) {
        setCurrentConversation(null);
        setMessages([]);
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    if (date.toDateString() === today.toDateString()) {
      return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    } else if (date.toDateString() === yesterday.toDateString()) {
      return 'Yesterday';
    } else {
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }
  };

  const renderMarkdown = (text: string) => {
    // Convert **bold** to <strong>
    const boldRegex = /\*\*(.*?)\*\*/g;
    const withBold = text.replace(boldRegex, '<strong>$1</strong>');
    
    // Convert numbered lists to proper HTML
    const lines = withBold.split('\n');
    const processedLines = lines.map(line => {
      // Convert numbered list items
      if (/^\d+\.\s/.test(line)) {
        return line.replace(/^(\d+\.\s)(.*)/, '<li>$2</li>');
      }
      // Convert bullet points
      if (/^\*\s/.test(line)) {
        return line.replace(/^\*\s(.*)/, '<li>$1</li>');
      }
      return line;
    });
    
    // Wrap consecutive list items in <ol> or <ul>
    let result = '';
    let inOrderedList = false;
    let inUnorderedList = false;
    
    for (let i = 0; i < processedLines.length; i++) {
      const line = processedLines[i];
      const isListItem = line.startsWith('<li>');
      const nextIsListItem = i < processedLines.length - 1 && processedLines[i + 1].startsWith('<li>');
      const prevWasNumbered = i > 0 && /^\d+\./.test(lines[i - 1]);
      const nextIsNumbered = i < lines.length - 1 && /^\d+\./.test(lines[i + 1]);
      
      if (isListItem) {
        const currentIsNumbered = /^\d+\./.test(lines[i]);
        
        if (!inOrderedList && !inUnorderedList) {
          if (currentIsNumbered) {
            result += '<ol>';
            inOrderedList = true;
          } else {
            result += '<ul>';
            inUnorderedList = true;
          }
        }
        
        result += line;
        
        if (!nextIsListItem) {
          if (inOrderedList) {
            result += '</ol>';
            inOrderedList = false;
          } else if (inUnorderedList) {
            result += '</ul>';
            inUnorderedList = false;
          }
        }
      } else {
        if (inOrderedList) {
          result += '</ol>';
          inOrderedList = false;
        } else if (inUnorderedList) {
          result += '</ul>';
          inUnorderedList = false;
        }
        result += line;
      }
      
      if (i < processedLines.length - 1) {
        result += '\n';
      }
    }
    
    return result;
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Sidebar - Chat History */}
      <div className={`${
        sidebarOpen ? 'w-72' : 'w-0'
      } transition-all duration-300 bg-slate-950/95 border-r border-slate-800/50 flex flex-col overflow-hidden backdrop-blur-xl`}>
        {/* Header with Branding */}
        <div className="p-4 border-b border-slate-800/50">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 via-blue-500 to-cyan-500 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <div>
              <h2 className="font-bold text-lg bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">VICTOR</h2>
              <p className="text-xs text-slate-500">AI Document Assistant</p>
            </div>
          </div>
          <button
            onClick={createNewChat}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 rounded-xl transition-all text-white font-semibold text-sm shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50 hover:scale-[1.02] active:scale-[0.98]"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New Conversation
          </button>
        </div>

        {/* Conversations List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {conversations.length === 0 ? (
            <div className="text-center text-slate-400 text-sm py-12">
              <svg className="w-12 h-12 mx-auto mb-3 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
              <p className="font-medium text-slate-500">No conversations</p>
              <p className="text-xs mt-1 text-slate-600">Create one to get started</p>
            </div>
          ) : (
            conversations.map(conv => (
              <div
                key={conv.conversation_id}
                onClick={() => {
                  setCurrentConversation(conv.conversation_id);
                  setCurrentTitle(conv.title);
                }}
                className={`p-3 rounded-xl cursor-pointer transition-all group ${
                  currentConversation === conv.conversation_id
                    ? 'bg-gradient-to-r from-blue-600/20 to-cyan-600/20 border border-blue-500/50 shadow-lg shadow-blue-500/10'
                    : 'bg-slate-900/50 border border-slate-800/50 hover:border-slate-700 hover:bg-slate-900/70 hover:shadow-lg'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className={`font-semibold text-sm truncate ${
                      currentConversation === conv.conversation_id
                        ? 'text-blue-300'
                        : 'text-slate-300'
                    }`}>
                      {conv.title}
                    </p>
                    <p className="text-xs text-slate-500 mt-1">
                      {formatDate(conv.updated_at)}
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteConversation(conv.conversation_id);
                    }}
                    className="p-1.5 rounded-lg opacity-0 group-hover:opacity-100 transition-all hover:bg-red-500/10 text-slate-500 hover:text-red-400"
                    title="Delete conversation"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Settings */}
        <div className="p-4 border-t border-slate-800/50 space-y-4 bg-slate-950/80">
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-slate-400">Temperature</label>
              <span className="text-xs font-mono text-cyan-400 bg-slate-900/50 px-2 py-1 rounded-lg border border-slate-800/50">
                {temperature.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
              style={{
                background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${temperature * 100}%, #1e293b ${temperature * 100}%, #1e293b 100%)`
              }}
            />
            <p className="text-xs text-slate-500 mt-1.5">Controls response creativity</p>
          </div>
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-slate-400">Documents (Top K)</label>
              <span className="text-xs font-mono text-cyan-400 bg-slate-900/50 px-2 py-1 rounded-lg border border-slate-800/50">
                {topK}
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
              style={{
                background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${(topK - 1) * 11.11}%, #1e293b ${(topK - 1) * 11.11}%, #1e293b 100%)`
              }}
            />
            <p className="text-xs text-slate-500 mt-1.5">Number of sources to retrieve</p>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-slate-900">
        {/* Top Bar */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800/50 bg-slate-950/80 backdrop-blur-xl">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-slate-800/50 rounded-lg transition-all text-slate-400 hover:text-cyan-400"
              title={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {sidebarOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
            <div>
              <h1 className="text-lg font-bold text-white">{currentTitle || 'New Conversation'}</h1>
              <p className="text-xs text-slate-500 mt-0.5">
                {messages.length > 0 ? `${messages.length} message${messages.length !== 1 ? 's' : ''}` : 'Ask anything about your documents'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 hidden sm:inline flex items-center gap-1.5">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
              LangChain RAG
            </span>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 bg-gradient-to-b from-slate-900 via-slate-900 to-slate-950">
          {messages.length === 0 && (
            <div className="h-full flex items-center justify-center">
              <div className="text-center max-w-md">
                <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-blue-600 to-cyan-500 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-500/30">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                  </svg>
                </div>
                <h2 className="text-xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">Ready to assist you</h2>
                <p className="text-slate-400">Ask questions about your documents and I'll provide answers based on the content</p>
              </div>
            </div>
          )}
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.map((message, idx) => {
              console.log(`Message ${idx}:`, message);
              console.log(`  - role: ${message.role}`);
              console.log(`  - content: "${message.content}"`);
              return (
                <div
                  key={message.message_id || `msg-${idx}`}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-2xl px-5 py-4 rounded-2xl shadow-lg ${
                      message.role === 'user'
                        ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-blue-500/20'
                        : 'bg-slate-900/80 text-slate-200 border border-slate-800/50 shadow-slate-900/50'
                    }`}
                  >
                    <div 
                      className="text-sm leading-relaxed whitespace-pre-wrap [&_strong]:font-semibold [&_ol]:list-decimal [&_ol]:ml-6 [&_ul]:list-disc [&_ul]:ml-6 [&_li]:my-1.5"
                      dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }}
                    />
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-slate-700/50 text-xs">
                        <p className="font-semibold mb-2 flex items-center gap-1.5 text-cyan-400">
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                          </svg>
                          Sources
                        </p>
                        <ul className="space-y-1.5">
                          {message.sources.slice(0, 3).map((source, i) => (
                            <li key={i} className={`flex items-start gap-2 ${
                              message.role === 'user' ? 'text-blue-100' : 'text-slate-400'
                            }`}>
                              <span className="text-cyan-500 mt-0.5">•</span>
                              <span>
                                <span className="font-medium text-cyan-400">{source.source_file}</span>
                                <span className="opacity-75"> (Page {source.page_idx})</span>
                                <span className="ml-1.5 opacity-60">– {(source.score * 100).toFixed(1)}% match</span>
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    <p className={`text-xs mt-2 opacity-60`}>
                      {new Date(message.created_at).toLocaleTimeString('en-US', { 
                        hour: '2-digit', 
                        minute: '2-digit' 
                      })}
                    </p>
                  </div>
                </div>
              );
            })}
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-slate-900/80 text-slate-200 px-5 py-4 rounded-2xl shadow-lg border border-slate-800/50">
                    <div className="flex items-center gap-3">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      </div>
                      <span className="text-sm text-slate-400">Analyzing documents...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-slate-800/50 p-6 bg-gradient-to-b from-slate-950/80 to-slate-950 backdrop-blur-xl">
          <div className="max-w-4xl mx-auto">
            <form onSubmit={sendMessage} className="flex gap-3">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                disabled={!currentConversation || loading}
                placeholder={currentConversation ? "Type your question here..." : "Please create or select a conversation first"}
                className="flex-1 px-4 py-3 bg-slate-900/80 border border-slate-800/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 disabled:opacity-50 disabled:cursor-not-allowed transition shadow-lg"
              />
              <button
                type="submit"
                disabled={!inputValue.trim() || !currentConversation || loading}
                className="px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-cyan-500 disabled:from-slate-800 disabled:to-slate-800 text-white rounded-xl transition font-medium disabled:cursor-not-allowed shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50 hover:scale-[1.02] active:scale-[0.98] disabled:shadow-none flex items-center gap-2"
                title="Send message"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    <span className="hidden sm:inline">Sending</span>
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                    <span className="hidden sm:inline">Send</span>
                  </>
                )}
              </button>
            </form>
            <div className="flex items-center justify-between mt-3 text-xs text-slate-500 dark:text-slate-500">
              <p className="flex items-center gap-1.5">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Adjust settings in the sidebar for different response styles
              </p>
              <p className="hidden sm:block">Press Enter to send</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
