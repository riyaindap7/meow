"use client";

import { useState } from "react";
import Link from "next/link";
import ThemeToggle from "@/components/ThemeToggle";
import { useTheme } from "@/lib/ThemeContext";

interface SearchResult {
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
}

interface RAGResponse {
  query: string;
  answer: string;
  sources: SearchResult[];
  model_used: string;
}

export default function Search() {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<RAGResponse | null>(null);
  const [message, setMessage] = useState("");
  const [selectedSourceIndex, setSelectedSourceIndex] = useState<number | null>(null);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();

    if (!query.trim()) {
      setMessage("Please enter a search query");
      return;
    }

    setSearching(true);
    setMessage("");
    console.log("Searching for:", query);

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          top_k: 10,
          temperature: 0.1,
        }),
      });

      console.log("Ask response status:", response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.log("Search error:", errorText);
        console.log("Response status:", response.status);
        
        if (response.status === 503) {
          setMessage("❌ Backend service unavailable. Please start the backend server.");
        } else if (response.status === 500) {
          try {
            const errorData = JSON.parse(errorText);
            setMessage(`❌ Server Error: ${errorData.detail || "Internal server error"}`);
          } catch {
            setMessage("❌ Internal server error. Check backend logs for details.");
          }
        } else if (response.status === 422) {
          try {
            const errorData = JSON.parse(errorText);
            console.log("Validation error details:", errorData);
            setMessage(`❌ Validation Error: ${JSON.stringify(errorData.detail)}`);
          } catch {
            setMessage(`❌ Validation Error (422): Invalid request format`);
          }
        } else if (response.status === 404) {
          setMessage("❌ Search endpoint not found. Please check backend is running.");
        } else {
          try {
            const errorData = JSON.parse(errorText);
            setMessage(`❌ Error (${response.status}): ${errorData.detail || errorText}`);
          } catch {
            setMessage(`❌ Error (${response.status}): ${errorText || "Unknown error"}`);
          }
        }
        setResults(null);
        return;
      }

      const data = await response.json();
      console.log("RAG Response:", data);
      
      if (data.sources && data.sources.length > 0) {
        console.log(`📚 Retrieved ${data.sources.length} relevant chunks:`);
        data.sources.forEach((source: any, idx: number) => {
          console.log(`[Chunk ${idx + 1}] ${source.text.substring(0, 150)}...`);
        });
      }

      if (data.answer) {
        setResults(data);
        setMessage(`✅ Answer generated using ${data.model_used}`);
      } else {
        setResults(null);
        setMessage("No answer generated. Please try again.");
      }
    } catch (err) {
      console.log("Ask error:", err);
      
      if (err instanceof TypeError) {
        if (err.message.includes("Failed to fetch")) {
          setMessage("❌ Cannot connect to backend. Is the server running on http://localhost:8000?");
        } else if (err.message.includes("NetworkError")) {
          setMessage("❌ Network error: Please check your connection and ensure backend is running");
        } else {
          setMessage(`❌ Network Error: ${err.message}`);
        }
      } else if (err instanceof Error) {
        setMessage(`❌ Error: ${err.message}`);
      } else {
        setMessage("❌ An unexpected error occurred. Please check the backend logs.");
      }
      
      setResults(null);
    } finally {
      setSearching(false);
    }
  }

  return (
    <main className={`min-h-screen transition-colors duration-300 ${
      isDark ? "bg-neutral-950 text-white" : "bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 text-gray-900"
    }`}>
      {/* Header with Navigation */}
      <header className={`border-b backdrop-blur sticky top-0 z-50 ${
        isDark ? "border-neutral-800 bg-neutral-900/50" : "border-gray-200 bg-gray-50/50"
      }`}>
        <div className="max-w-6xl mx-auto px-6 py-4 flex justify-between items-center">
          <div>
            <h1 className={`text-2xl font-bold ${isDark ? "text-cyan-400" : "text-blue-600"}`}>Victor</h1>
            <p className={`text-xs ${isDark ? "text-gray-500" : "text-gray-600"} mt-1`}>Document Search & Analysis</p>
          </div>
          <div className="flex gap-3 items-center">
            <ThemeToggle />
            <Link
              href="/"
              className={`text-sm transition-colors ${
                isDark ? "text-gray-400 hover:text-cyan-400" : "text-gray-600 hover:text-blue-600"
              }`}
            >
              ← Back to Home
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 py-16">
        {/* Search Section */}
        <section className={`mb-16 rounded-3xl p-12 transition-all ${
          isDark
            ? "bg-gradient-to-br from-cyan-950/40 to-blue-950/40 border-2 border-cyan-600/30 shadow-2xl shadow-cyan-500/10"
            : "bg-gradient-to-br from-blue-100/60 to-indigo-100/40 border-2 border-blue-300 shadow-2xl shadow-blue-500/20"
        }`}>
          <div className="space-y-8">
            {/* Title */}
            <div>
              <h2 className={`text-5xl font-black mb-3 bg-clip-text text-transparent ${
                isDark
                  ? "bg-gradient-to-r from-cyan-300 to-blue-400"
                  : "bg-gradient-to-r from-blue-600 to-indigo-600"
              }`}>Ask a Question</h2>
              <p className={`text-lg ${isDark ? "text-gray-300" : "text-gray-700"}`}>Search across all your uploaded documents with AI-powered intelligence</p>
            </div>

            {/* Search Input */}
            <form onSubmit={handleSearch} className="flex gap-3 items-center">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What do you want to know?"
                className={`flex-1 px-6 py-4 text-lg rounded-xl focus:outline-none focus:ring-2 transition-all font-medium ${
                  isDark
                    ? "bg-neutral-900/80 border-2 border-cyan-600/50 text-white placeholder-gray-500 focus:border-cyan-400 focus:ring-cyan-500/30 shadow-lg"
                    : "bg-white/90 border-2 border-blue-400 text-gray-900 placeholder-gray-500 focus:border-blue-600 focus:ring-blue-500/30 shadow-lg"
                }`}
                disabled={searching}
              />
              <button
                type="submit"
                disabled={searching || !query.trim()}
                className={`px-10 py-4 rounded-xl font-bold text-lg transition-all shadow-lg ${
                  isDark
                    ? "bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:from-neutral-700 disabled:to-neutral-700 text-white hover:shadow-xl hover:shadow-cyan-500/40"
                    : "bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 disabled:from-gray-400 disabled:to-gray-400 text-white hover:shadow-xl hover:shadow-blue-500/40"
                }`}
              >
                {searching ? "🔄 Analyzing..." : "🔍 Search"}
              </button>
            </form>

            {/* Status Message */}
            {message && (
              <div className={`px-6 py-4 rounded-xl text-base font-semibold border-2 backdrop-blur-sm ${
                message.startsWith("✅")
                  ? isDark
                    ? "bg-emerald-900/40 text-emerald-200 border-emerald-600 shadow-lg shadow-emerald-500/20"
                    : "bg-green-100 text-green-800 border-green-400 shadow-lg shadow-green-500/20"
                  : isDark
                  ? "bg-red-900/40 text-red-200 border-red-600 shadow-lg shadow-red-500/20"
                  : "bg-red-100 text-red-800 border-red-400 shadow-lg shadow-red-500/20"
              }`}>
                {message}
              </div>
            )}
          </div>
        </section>

        {/* Results Section */}
        {results && (
          <section className="space-y-8">
            {/* AI Answer */}
            <div className="border border-cyan-800/50 rounded-lg p-8 bg-gradient-to-br from-cyan-900/10 to-blue-900/10">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <h3 className="text-2xl font-bold text-cyan-300 mb-1">Answer</h3>
                  <p className="text-xs text-gray-500">Generated by {results.model_used}</p>
                </div>
              </div>
              <div className="text-gray-100 leading-relaxed whitespace-pre-wrap text-base">
                {results.answer}
              </div>
            </div>

            {/* Sources */}
            <div>
              <div className="mb-6">
                <h3 className="text-2xl font-bold mb-2">Referenced Sources</h3>
                <p className="text-gray-500 text-sm">{results.sources?.length || 0} document chunks used</p>
              </div>

              <div className="grid gap-4">
                {results.sources?.map((source, index) => (
                  <article
                    key={index}
                    className="border border-neutral-700 rounded-lg p-5 hover:border-cyan-600/50 hover:bg-neutral-900/30 transition-all duration-200"
                  >
                    {/* Source Header */}
                    <div className="flex items-start justify-between mb-4 pb-4 border-b border-neutral-800">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="inline-block bg-cyan-900/40 text-cyan-300 px-3 py-1 rounded text-xs font-bold border border-cyan-700/50">
                            #{index + 1}
                          </span>
                          <span className="text-xs text-gray-500">•</span>
                          <span className="text-sm font-medium text-cyan-400">{source.source_file}</span>
                        </div>
                        <div className="flex gap-4 flex-wrap">
                          <div className="flex items-center gap-1 text-xs text-gray-400">
                            <span className="text-blue-400">📄</span>
                            <span>Page {source.page_idx}</span>
                          </div>
                          <div className={`flex items-center gap-1 text-xs font-semibold ${
                            source.score > 0.7 ? 'text-emerald-400' :
                            source.score > 0.5 ? 'text-cyan-400' : 'text-amber-400'
                          }`}>
                            <span>🎯</span>
                            <span>Match: {(source.score * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Source Content */}
                    <div className="text-gray-300 text-sm leading-relaxed bg-neutral-900/40 rounded border-l-2 border-cyan-600/50 pl-4 py-3 mb-4">
                      {source.text}
                    </div>

                    {/* Action Button */}
                    <div className="flex gap-2">
                      <button
                        onClick={() => setSelectedSourceIndex(index)}
                        className="text-xs px-4 py-2 bg-cyan-900/30 hover:bg-cyan-900/50 text-cyan-400 rounded border border-cyan-700/50 transition-colors font-medium"
                      >
                        📖 View PDF & Highlight
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            </div>
          </section>
        )}

        {/* Empty State */}
        {!results && !searching && (
          <div className="text-center py-16">
            <div className="text-5xl mb-4">🔍</div>
            <h3 className="text-xl font-semibold text-gray-300 mb-2">Ready to search</h3>
            <p className="text-gray-500">Enter a question above to find relevant information in your documents</p>
          </div>
        )}

        {/* Loading State */}
        {searching && (
          <div className="text-center py-16">
            <div className="inline-block">
              <div className="animate-spin text-4xl mb-4">⚙️</div>
              <h3 className="text-xl font-semibold text-gray-300 mb-2">Searching documents</h3>
              <p className="text-gray-500 text-sm">This may take a moment...</p>
            </div>
          </div>
        )}
      </div>

      {/* PDF Viewer Modal */}
      {selectedSourceIndex !== null && results && results.sources[selectedSourceIndex] && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-neutral-900 rounded-lg w-full max-w-4xl max-h-[90vh] flex flex-col border border-neutral-700 shadow-2xl">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-6 border-b border-neutral-700">
              <div className="flex-1">
                <h2 className="text-2xl font-bold text-cyan-400 mb-1">
                  {results.sources[selectedSourceIndex].source_file}
                </h2>
                <p className="text-sm text-gray-400">
                  📄 Page {results.sources[selectedSourceIndex].page_idx} • 
                  🎯 Relevance: {(results.sources[selectedSourceIndex].score * 100).toFixed(1)}%
                </p>
              </div>
              <button
                onClick={() => setSelectedSourceIndex(null)}
                className="text-gray-400 hover:text-cyan-400 text-2xl transition-colors"
                aria-label="Close PDF viewer"
              >
                ✕
              </button>
            </div>

            {/* Modal Content */}
            <div className="flex-1 overflow-y-auto p-6">
              <div className="bg-neutral-950 rounded p-4 border border-neutral-700">
                {/* Referenced Text Highlight */}
                <div className="mb-6">
                  <h3 className="text-sm font-semibold text-cyan-400 uppercase tracking-wide mb-3">
                    Referenced Text
                  </h3>
                  <div className="bg-neutral-900/50 border-l-4 border-cyan-500 p-4 rounded text-gray-200 leading-relaxed text-sm whitespace-pre-wrap font-mono">
                    {results.sources[selectedSourceIndex].text}
                  </div>
                </div>

                {/* Note */}
                <div className="flex items-start gap-3 p-3 bg-cyan-900/20 border border-cyan-700/30 rounded text-xs text-gray-300">
                  <span className="text-cyan-400 mt-0.5">ℹ️</span>
                  <p>This is the exact text passage extracted from page {results.sources[selectedSourceIndex].page_idx} of the document. Full PDF viewing coming soon.</p>
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="border-t border-neutral-700 p-4 flex gap-3 justify-end bg-neutral-900/50">
              <button
                onClick={() => setSelectedSourceIndex(null)}
                className="px-4 py-2 bg-neutral-700 hover:bg-neutral-600 text-white rounded transition-colors font-medium"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="border-t border-neutral-800 bg-neutral-900/50 mt-16 py-6">
        <div className="max-w-6xl mx-auto px-6 text-center text-sm text-gray-500">
          <p>&copy; 2024 Victor. Powered by Milvus Vector Search & OpenRouter LLM</p>
        </div>
      </footer>
    </main>
  );
}
