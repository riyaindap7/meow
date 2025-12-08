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
    <main
      data-theme={theme}
      className="min-h-screen bg-black text-gray-100 transition-colors duration-300"
    >
      {/* Header with Navigation */}
      <header className="border-b border-neutral-900 bg-neutral-950/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-100">Victor</h1>
            <p className="text-xs text-gray-500 mt-1">Document Search & Analysis</p>
          </div>
          <div className="flex gap-3 items-center">
            <ThemeToggle />
            <Link
              href="/"
              className="text-sm text-gray-400 hover:text-gray-100 transition-colors"
            >
              ← Back to Home
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 py-16">
        {/* Search Section */}
        <section className="mb-16 rounded-3xl p-12 bg-neutral-950 border border-neutral-900 shadow-2xl shadow-black/60">
          <div className="space-y-8">
            {/* Title */}
            <div>
              <h2 className="text-5xl font-black mb-3 text-gray-50">
                Ask a Question
              </h2>
              <p className="text-lg text-gray-400">
                Search across all your uploaded documents with AI-powered intelligence
              </p>
            </div>

            {/* Search Input */}
            <form onSubmit={handleSearch} className="flex gap-3 items-center">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What do you want to know?"
                className="flex-1 px-6 py-4 text-lg rounded-xl bg-neutral-900 border border-neutral-700 text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-gray-400 shadow-lg"
                disabled={searching}
              />
              <button
                type="submit"
                disabled={searching || !query.trim()}
                className="px-10 py-4 rounded-xl font-bold text-lg bg-neutral-800 hover:bg-neutral-700 disabled:bg-neutral-900 text-gray-100 transition-all shadow-lg disabled:cursor-not-allowed"
              >
                {searching ? "🔄 Analyzing..." : "Search"}
              </button>
            </form>

            {/* Status Message */}
            {message && (
              <div
                className={`px-6 py-4 rounded-xl text-base font-semibold border backdrop-blur-sm ${
                  message.startsWith("✅")
                    ? "bg-neutral-900 text-gray-100 border-neutral-600 shadow-lg shadow-black/40"
                    : "bg-neutral-950 text-gray-200 border-neutral-700 shadow-lg shadow-black/50"
                }`}
              >
                {message}
              </div>
            )}
          </div>
        </section>

        {/* Results Section */}
        {results && (
          <section className="space-y-8">
            {/* AI Answer */}
            <div className="border border-neutral-800 rounded-xl p-8 bg-neutral-950/80">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <h3 className="text-2xl font-bold text-gray-100 mb-1">Answer</h3>
                  <p className="text-xs text-gray-500">
                    Generated by {results.model_used}
                  </p>
                </div>
              </div>
              <div className="text-gray-200 leading-relaxed whitespace-pre-wrap text-base">
                {results.answer}
              </div>
            </div>

            {/* Sources */}
            <div>
              <div className="mb-6">
                <h3 className="text-2xl font-bold mb-2 text-gray-100">
                  Referenced Sources
                </h3>
                <p className="text-gray-500 text-sm">
                  {results.sources?.length || 0} document chunks used
                </p>
              </div>

              <div className="grid gap-4">
                {results.sources?.map((source, index) => (
                  <article
                    key={index}
                    className="border border-neutral-800 rounded-lg p-5 hover:border-neutral-500 hover:bg-neutral-950 transition-all duration-200"
                  >
                    {/* Source Header */}
                    <div className="flex items-start justify-between mb-4 pb-4 border-b border-neutral-900">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="inline-block bg-neutral-900 text-gray-200 px-3 py-1 rounded text-xs font-bold border border-neutral-700">
                            #{index + 1}
                          </span>
                          <span className="text-xs text-gray-600">•</span>
                          <span className="text-sm font-medium text-gray-200">
                            {source.source_file}
                          </span>
                        </div>
                        <div className="flex gap-4 flex-wrap">
                          <div className="flex items-center gap-1 text-xs text-gray-500">
                            <span></span>
                            <span>Page {source.page_idx}</span>
                          </div>
                          <div className="flex items-center gap-1 text-xs font-semibold text-gray-400">
                            <span></span>
                            <span>
                              Match: {(source.score * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Source Content */}
                    <div className="text-gray-300 text-sm leading-relaxed bg-neutral-950 rounded border-l-2 border-neutral-600 pl-4 py-3 mb-4">
                      {source.text}
                    </div>

                    {/* Action Button */}
                    <div className="flex gap-2">
                      <button
                        onClick={() => setSelectedSourceIndex(index)}
                        className="text-xs px-4 py-2 bg-neutral-900 hover:bg-neutral-800 text-gray-200 rounded border border-neutral-700 transition-colors font-medium"
                      >
                        View PDF & Highlight
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
            <div className="text-5xl mb-4"></div>
            <h3 className="text-xl font-semibold text-gray-100 mb-2">
              Ready to search
            </h3>
            <p className="text-gray-500">
              Enter a question above to find relevant information in your
              documents
            </p>
          </div>
        )}

        {/* Loading State */}
        {searching && (
          <div className="text-center py-16">
            <div className="inline-block">
              <div className="animate-spin text-4xl mb-4">⚙️</div>
              <h3 className="text-xl font-semibold text-gray-100 mb-2">
                Searching documents
              </h3>
              <p className="text-gray-500 text-sm">
                This may take a moment...
              </p>
            </div>
          </div>
        )}
      </div>

      {/* PDF Viewer Modal */}
      {selectedSourceIndex !== null &&
        results &&
        results.sources[selectedSourceIndex] && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-neutral-950 rounded-lg w-full max-w-4xl max-h-[90vh] flex flex-col border border-neutral-800 shadow-2xl shadow-black">
              {/* Modal Header */}
              <div className="flex items-center justify-between p-6 border-b border-neutral-800">
                <div className="flex-1">
                  <h2 className="text-2xl font-bold text-gray-100 mb-1">
                    {results.sources[selectedSourceIndex].source_file}
                  </h2>
                  <p className="text-sm text-gray-500">
                    📄 Page {results.sources[selectedSourceIndex].page_idx} • 
                    Relevance:{" "}
                    {(
                      results.sources[selectedSourceIndex].score * 100
                    ).toFixed(1)}
                    %
                  </p>
                </div>
                <button
                  onClick={() => setSelectedSourceIndex(null)}
                  className="text-gray-500 hover:text-gray-200 text-2xl transition-colors"
                  aria-label="Close PDF viewer"
                >
                  ✕
                </button>
              </div>

              {/* Modal Content */}
              <div className="flex-1 overflow-y-auto p-6">
                <div className="bg-black rounded p-4 border border-neutral-800">
                  {/* Referenced Text Highlight */}
                  <div className="mb-6">
                    <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-3">
                      Referenced Text
                    </h3>
                    <div className="bg-neutral-950 border-l-4 border-neutral-600 p-4 rounded text-gray-100 leading-relaxed text-sm whitespace-pre-wrap font-mono">
                      {results.sources[selectedSourceIndex].text}
                    </div>
                  </div>

                  {/* Note */}
                  <div className="flex items-start gap-3 p-3 bg-neutral-900 border border-neutral-800 rounded text-xs text-gray-300">
                    <span className="mt-0.5">ℹ️</span>
                    <p>
                      This is the exact text passage extracted from page{" "}
                      {results.sources[selectedSourceIndex].page_idx} of the
                      document. Full PDF viewing coming soon.
                    </p>
                  </div>
                </div>
              </div>

              {/* Modal Footer */}
              <div className="border-t border-neutral-800 p-4 flex gap-3 justify-end bg-neutral-950">
                <button
                  onClick={() => setSelectedSourceIndex(null)}
                  className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-gray-100 rounded transition-colors font-medium"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}

      {/* Footer */}
      <footer className="border-t border-neutral-900 bg-black mt-16 py-6">
        <div className="max-w-6xl mx-auto px-6 text-center text-sm text-gray-600">
          <p>&copy; 2024 Victor. Powered by Milvus Vector Search &amp; OpenRouter LLM</p>
        </div>
      </footer>
    </main>
  );
}
