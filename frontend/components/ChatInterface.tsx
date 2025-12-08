'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import VoiceInput from './VoiceInput';

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

  // Voice input handler
  const handleVoiceTranscript = (transcript: string) => {
    setInputValue(transcript);
  };

  const handleVoiceError = (error: string) => {
    console.error('Voice input error:', error);
    // You could add a toast notification here
  };

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
    // First, escape HTML to prevent XSS, but preserve our own tags
    let processedText = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    
    // Convert headers (must be at start of line)
    processedText = processedText.replace(/^### (.+)$/gm, '<h3 class="text-lg font-semibold mt-6 mb-3 text-blue-400 border-b border-blue-800 pb-1">$1</h3>');
    processedText = processedText.replace(/^## (.+)$/gm, '<h2 class="text-xl font-bold mt-8 mb-4 text-blue-300 border-b border-blue-700 pb-2">$1</h2>');
    processedText = processedText.replace(/^# (.+)$/gm, '<h1 class="text-2xl font-bold mt-8 mb-5 text-blue-200 border-b border-blue-600 pb-2">$1</h1>');
    
    // Convert **bold** and __bold__ to <strong>
    processedText = processedText.replace(/(\*\*|__)((?:(?!\1).)+)\1/g, '<strong class="font-semibold text-white bg-slate-800 px-1 rounded">$2</strong>');
    
    // Convert *italic* and _italic_ to <em> (but not when part of bold)
    processedText = processedText.replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, '<em class="italic text-blue-200">$1</em>');
    processedText = processedText.replace(/(?<!_)_([^_\n]+)_(?!_)/g, '<em class="italic text-blue-200">$1</em>');
    
    // Convert `inline code` to styled code
    processedText = processedText.replace(/`([^`\n]+)`/g, '<code class="bg-slate-800 text-green-300 px-2 py-1 rounded font-mono text-sm border border-slate-600">$1</code>');
    
    // Convert ```code blocks``` to styled blocks
    processedText = processedText.replace(/```([^`\n]*)\n([\s\S]*?)```/g, 
      '<pre class="bg-slate-900 text-green-300 p-4 rounded-lg border border-slate-600 my-4 overflow-x-auto"><code class="font-mono text-sm">$2</code></pre>');
    
    // Convert links [text](url) to clickable links
    processedText = processedText.replace(/\[([^\]]+)\]\(([^)]+)\)/g, 
      '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-blue-400 hover:text-blue-300 underline">$1</a>');
    
    // Handle blockquotes (lines starting with >)
    processedText = processedText.replace(/^&gt;\s*(.+)$/gm, 
      '<blockquote class="border-l-4 border-blue-500 pl-4 my-3 text-slate-300 italic bg-slate-800/50 py-2">$1</blockquote>');
    
    // Handle horizontal rules (--- or ***)
    processedText = processedText.replace(/^(---|\*\*\*)$/gm, 
      '<hr class="border-t border-slate-600 my-6" />');
    
    // Handle tables and complex formatting
    const lines = processedText.split('\n');
    let result = '';
    let inTable = false;
    let inOrderedList = false;
    let inUnorderedList = false;
    let inCodeBlock = false;
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmedLine = line.trim();
      
      // Skip empty lines in special contexts
      if (!trimmedLine && (inTable || inOrderedList || inUnorderedList)) {
        // Close current context on empty line
        if (inTable) {
          result += '</tbody></table>';
          inTable = false;
        }
        if (inOrderedList) {
          result += '</ol>';
          inOrderedList = false;
        }
        if (inUnorderedList) {
          result += '</ul>';
          inUnorderedList = false;
        }
        result += '<br class="my-2" />';
        continue;
      }
      
      // Handle table rows
      if (trimmedLine.includes('|') && trimmedLine.split('|').length > 2) {
        const cells = trimmedLine.split('|').map(cell => cell.trim()).filter(cell => cell);
        
        // Skip table separator lines (like |:--|:--|)
        if (cells.every(cell => cell.match(/^:?-+:?$/))) {
          continue;
        }
        
        if (!inTable) {
          result += `<table class="w-full border-collapse border border-slate-600 my-6 rounded-lg overflow-hidden">
                     <tbody class="divide-y divide-slate-600">`;
          inTable = true;
        }
        
        result += '<tr class="hover:bg-slate-800/30">';
        cells.forEach((cell, index) => {
          const isFirstRow = result.indexOf('<tr') === result.lastIndexOf('<tbody') + 7;
          const cellClass = isFirstRow 
            ? "bg-slate-700 font-semibold text-blue-200 px-4 py-3 text-left border-r border-slate-600 last:border-r-0" 
            : "px-4 py-3 text-slate-200 border-r border-slate-600 last:border-r-0";
          result += `<td class="${cellClass}">${cell}</td>`;
        });
        result += '</tr>';
        continue;
      }
      
      // Close table if we're no longer in one
      if (inTable) {
        result += '</tbody></table>';
        inTable = false;
      }
      
      // Handle ordered lists (1. 2. 3.)
      const orderedMatch = trimmedLine.match(/^(\d+)\.\s+(.+)$/);
      if (orderedMatch) {
        if (!inOrderedList) {
          if (inUnorderedList) {
            result += '</ul>';
            inUnorderedList = false;
          }
          result += '<ol class="list-none space-y-2 my-4 ml-4">';
          inOrderedList = true;
        }
        result += `<li class="flex items-start">
                    <span class="font-bold text-blue-400 min-w-[2rem] mr-2">${orderedMatch[1]}.</span>
                    <span class="text-slate-200">${orderedMatch[2]}</span>
                   </li>`;
        continue;
      }
      
      // Handle unordered lists (- * +)
      const unorderedMatch = trimmedLine.match(/^[\*\-\+]\s+(.+)$/);
      if (unorderedMatch) {
        if (!inUnorderedList) {
          if (inOrderedList) {
            result += '</ol>';
            inOrderedList = false;
          }
          result += '<ul class="list-none space-y-2 my-4 ml-4">';
          inUnorderedList = true;
        }
        result += `<li class="flex items-start">
                    <span class="text-blue-400 min-w-[1rem] mr-2">•</span>
                    <span class="text-slate-200">${unorderedMatch[1]}</span>
                   </li>`;
        continue;
      }
      
      // Close lists if we're no longer in them
      if (inOrderedList && !orderedMatch) {
        result += '</ol>';
        inOrderedList = false;
      }
      if (inUnorderedList && !unorderedMatch) {
        result += '</ul>';
        inUnorderedList = false;
      }
      
      // Handle indented content (4+ spaces or tab)
      if (trimmedLine && (line.startsWith('    ') || line.startsWith('\t'))) {
        result += `<div class="ml-8 my-2 p-3 bg-slate-800/40 rounded border-l-2 border-blue-600 text-slate-300 font-mono text-sm">${trimmedLine}</div>`;
        continue;
      }
      
      // Handle task lists - [x] and [ ]
      const taskMatch = trimmedLine.match(/^[\*\-\+]?\s*\[([ x])\]\s+(.+)$/);
      if (taskMatch) {
        const checked = taskMatch[1] === 'x';
        const checkboxClass = checked ? 'text-green-400' : 'text-slate-500';
        const textClass = checked ? 'line-through text-slate-400' : 'text-slate-200';
        result += `<div class="flex items-start my-2 ml-4">
                    <span class="${checkboxClass} mr-2">${checked ? '☑' : '☐'}</span>
                    <span class="${textClass}">${taskMatch[2]}</span>
                   </div>`;
        continue;
      }
      
      // Regular paragraphs and text
      if (trimmedLine) {
        // Check if this line contains special formatting that wasn't caught above
        if (trimmedLine.includes('&lt;') && trimmedLine.includes('&gt;')) {
          // This might be HTML that got escaped, preserve it
          result += `<div class="my-3 leading-relaxed text-slate-200">${trimmedLine}</div>`;
        } else {
          result += `<p class="my-3 leading-relaxed text-slate-200">${trimmedLine}</p>`;
        }
      } else {
        result += '<br class="my-2" />';
      }
    }
    
    // Close any remaining open tags
    if (inTable) {
      result += '</tbody></table>';
    }
    if (inOrderedList) {
      result += '</ol>';
    }
    if (inUnorderedList) {
      result += '</ul>';
    }
    
    // Final cleanup and optimization
    result = result
      .replace(/(<br[^>]*>[\s]*){3,}/g, '<br class="my-4" />') // Collapse multiple breaks
      .replace(/(<p[^>]*>[\s]*<\/p>[\s]*)+/g, '<br class="my-2" />') // Remove empty paragraphs
      .replace(/&amp;/g, '&') // Restore ampersands in our generated HTML
      .replace(/&lt;(\/?(?:strong|em|code|pre|h[1-6]|table|tbody|tr|td|ol|ul|li|br|hr|blockquote|a)[^>]*)&gt;/g, '<$1>'); // Restore our HTML tags
    
    return result;
  };

  return (
    <div className="flex h-screen bg-black">
      {/* Sidebar - Chat History */}
      <div className={`${
        sidebarOpen ? 'w-72' : 'w-0'
      } transition-all duration-300 bg-black border-r border-gray-800 flex flex-col overflow-hidden`}>
        {/* Header with Branding */}
        <div className="p-4 border-b border-gray-800">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-300 via-blue-200 to-cyan-200 rounded-xl flex items-center justify-center shadow-lg shadow-blue-200/20">
              <svg className="w-6 h-6 text-gray-800" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-blue-300 to-blue-200 hover:from-blue-200 hover:to-cyan-200 rounded-xl transition-all text-gray-800 font-semibold text-sm shadow-lg shadow-blue-200/30 hover:shadow-blue-200/50 hover:scale-[1.02] active:scale-[0.98]"
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
                    ? 'bg-blue-200/30 border border-blue-300/50 shadow-lg shadow-blue-200/10'
                    : 'bg-gray-900/50 border border-gray-800 hover:border-gray-700 hover:bg-gray-900/70 hover:shadow-lg'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className={`font-semibold text-sm truncate ${
                      currentConversation === conv.conversation_id
                        ? 'text-blue-600'
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
        {/* <div className="p-4 border-t border-slate-800/50 space-y-4 bg-slate-950/80">
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
        </div> */}
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-black">
        {/* Top Bar */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-black">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-gray-800 rounded-lg transition-all text-gray-400 hover:text-cyan-300"
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
        <div className="flex-1 overflow-y-auto p-6 bg-black">
          {messages.length === 0 && (
            <div className="h-full flex items-center justify-center">
              <div className="text-center max-w-md">
                <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-blue-300 to-cyan-200 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-200/30">
                  <svg className="w-8 h-8 text-gray-800" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                  </svg>
                </div>
                <h2 className="text-xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">Ready to assist you</h2>
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
                        ? 'bg-gradient-to-r from-blue-300 to-blue-200 text-gray-800 shadow-blue-200/20'
                        : 'bg-gray-900 text-gray-200 border border-gray-700 shadow-gray-900/50'
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
                              message.role === 'user' ? 'text-gray-700' : 'text-slate-400'
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
                  <div className="bg-gray-900 text-gray-200 px-5 py-4 rounded-2xl shadow-lg border border-gray-700">
                    <div className="flex items-center gap-3">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      </div>
                      <span className="text-sm text-gray-400">Analyzing documents...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-800 p-6 bg-black">
          <div className="max-w-4xl mx-auto">
            <form onSubmit={sendMessage} className="flex gap-3 items-end">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                disabled={!currentConversation || loading}
                placeholder={currentConversation ? "Type your question here..." : "Please create or select a conversation first"}
                className="flex-1 px-4 py-3 bg-gray-900 border border-gray-700 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 disabled:opacity-50 disabled:cursor-not-allowed transition shadow-lg"
              />
              
              {/* Voice Input */}
              <VoiceInput
                onTranscript={handleVoiceTranscript}
                onError={handleVoiceError}
                apiBaseUrl={API_URL}
                disabled={!currentConversation || loading}
                isDark={true}
              />
              
              <button
                type="submit"
                disabled={!inputValue.trim() || !currentConversation || loading}
                className="px-6 py-3 bg-gradient-to-r from-blue-300 to-blue-200 hover:from-blue-200 hover:to-cyan-200 disabled:from-gray-800 disabled:to-gray-800 text-gray-800 rounded-xl transition font-medium disabled:cursor-not-allowed shadow-lg shadow-blue-200/30 hover:shadow-blue-200/50 hover:scale-[1.02] active:scale-[0.98] disabled:shadow-none flex items-center gap-2"
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
            <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
              <p className="flex items-center gap-1.5">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Adjust settings in the sidebar for different response styles
              </p>
              <div className="hidden sm:flex items-center gap-4 text-xs">
                <p>🎤 Click mic to speak</p>
                <p>Press Enter to send</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
