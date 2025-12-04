"use client";

import { useState } from "react";
import Link from "next/link";

interface SearchResult {
  id: string;
  filename: string;
  excerpt: string;
  score: number;
  local_path: string;
}

export default function Search() {
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [message, setMessage] = useState("");
  const [hasSearched, setHasSearched] = useState(false);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();

    if (!query.trim()) {
      setMessage("Please enter a search query");
      return;
    }

    setSearching(true);
    setMessage("");
    setHasSearched(true);
    console.log("Searching for:", query);

    try {
      const response = await fetch("http://localhost:8000/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          top_k: 10,
        }),
      });

      console.log("Search response status:", response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.log("Search error:", errorText);
        
        // Handle different HTTP status codes
        if (response.status === 503) {
          setMessage("❌ Backend service unavailable. Please start the backend server.");
        } else if (response.status === 500) {
          try {
            const errorData = JSON.parse(errorText);
            setMessage(`❌ Server Error: ${errorData.detail || "Internal server error"}`);
          } catch {
            setMessage("❌ Internal server error. Check backend logs for details.");
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
        setResults([]);
        return;
      }

      const data = await response.json();
      console.log("Search results:", data);

      if (data.results && data.results.length > 0) {
        setResults(data.results);
        setMessage(`Found ${data.results.length} result(s)`);
      } else {
        setResults([]);
        setMessage("No documents found matching your query");
      }
    } catch (err) {
      console.log("Search error:", err);
      
      // Provide specific error messages for common issues
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
      
      setResults([]);
    } finally {
      setSearching(false);
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      {/* Navigation */}
      <nav className="flex justify-between items-center px-8 py-6 bg-white bg-opacity-80 backdrop-blur-md shadow-sm">
        <Link
          href="/"
          className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600"
        >
          Victor
        </Link>
        <div className="space-x-6 flex items-center">
          <Link
            href="/upload"
            className="text-gray-600 hover:text-blue-600 transition"
          >
            Upload
          </Link>
          <Link
            href="/"
            className="text-gray-600 hover:text-blue-600 transition font-semibold"
          >
            Back to Home
          </Link>
        </div>
      </nav>

      {/* Search Section */}
      <section className="max-w-4xl mx-auto px-8 py-16">
        <div className="space-y-8">
          {/* Header */}
          <div className="text-center space-y-4">
            <h1 className="text-5xl font-bold text-gray-900">
              Search Your Documents
            </h1>
            <p className="text-xl text-gray-600">
              Find information across all your uploaded PDFs
            </p>
          </div>

          {/* Search Form */}
          <form
            onSubmit={handleSearch}
            className="bg-white rounded-2xl shadow-xl p-8 space-y-6"
          >
            <div className="space-y-3">
              <label className="block text-sm font-semibold text-gray-700">
                Enter your search query
              </label>
              <div className="flex gap-3">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g., What are the key benefits?"
                  className="flex-1 px-6 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 transition text-black placeholder-gray-400"
                  disabled={searching}
                />
                <button
                  type="submit"
                  disabled={searching}
                  className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:shadow-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {searching ? "Searching..." : "Search"}
                </button>
              </div>
            </div>

            {/* Status Message */}
            {message && (
              <div
                className={`p-4 rounded-lg text-sm font-medium ${
                  message.startsWith("Found") || message.startsWith("Network")
                    ? "bg-green-50 text-green-700 border border-green-200"
                    : "bg-yellow-50 text-yellow-700 border border-yellow-200"
                }`}
              >
                {message}
              </div>
            )}
          </form>

          {/* Results Section */}
          {hasSearched && (
            <div className="space-y-4">
              <h2 className="text-2xl font-bold text-gray-900">
                Results ({results.length})
              </h2>

              {results.length > 0 ? (
                <div className="space-y-4">
                  {results.map((result, index) => (
                    <div
                      key={index}
                      className="bg-white rounded-xl border-2 border-gray-200 hover:border-blue-400 hover:shadow-lg transition p-6 space-y-3"
                    >
                      <div className="flex justify-between items-start gap-4">
                        <div className="flex-1">
                          <h3 className="text-lg font-bold text-gray-900">
                            {result.filename}
                          </h3>
                          <p className="text-sm text-gray-500 mt-1">
                            Match Score: {(result.score * 100).toFixed(1)}%
                          </p>
                        </div>
                        <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-semibold">
                          #{index + 1}
                        </span>
                      </div>

                      <p className="text-gray-700 line-clamp-3">
                        {result.excerpt}
                      </p>

                      <div className="flex gap-3 pt-2">
                        <button className="px-4 py-2 bg-blue-50 text-blue-600 rounded-lg text-sm font-semibold hover:bg-blue-100 transition">
                          View Details
                        </button>
                        <button className="px-4 py-2 bg-gray-50 text-gray-600 rounded-lg text-sm font-semibold hover:bg-gray-100 transition">
                          Download
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="bg-white rounded-xl border-2 border-dashed border-gray-300 p-12 text-center">
                  <p className="text-gray-500 text-lg">
                    No documents found matching your search
                  </p>
                  <p className="text-gray-400 mt-2">
                    Try uploading more documents or refine your search query
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-gray-400 py-12 mt-20">
        <div className="max-w-6xl mx-auto px-8 text-center">
          <p>&copy; 2024 Victor. All rights reserved.</p>
        </div>
      </footer>
    </main>
  );
}
