"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import ThemeToggle from "@/components/ThemeToggle";
import { useTheme } from "@/lib/ThemeContext";

// Use browser's window object for API URL, fallback to localhost
const getApiUrl = () => {
  if (typeof window !== 'undefined') {
    return process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  }
  return "http://localhost:8000";
};

interface Collection {
  name: string;
  num_entities?: number;
}

interface CollectionDetails {
  name: string;
  description: string;
  num_entities: number;
  schema: {
    fields: Array<{
      name: string;
      type: string;
      is_primary: boolean;
      auto_id: boolean;
      dim?: number;
      max_length?: number;
    }>;
  };
  indexes: Array<{
    field: string;
    index_type: string;
    metric_type: string;
  }>;
}

interface ServerStatus {
  status: string;
  host: string;
  port: string;
  collections_count: number;
  total_entities: number;
  error?: string;
}

export default function MilvusAdmin() {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  const [collections, setCollections] = useState<Collection[]>([]);
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null);
  const [collectionDetails, setCollectionDetails] = useState<CollectionDetails | null>(null);
  const [queryResults, setQueryResults] = useState<any>(null);
  const [serverStatus, setServerStatus] = useState<ServerStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"collections" | "explorer" | "stats">("collections");
  const [message, setMessage] = useState("");
  const [mounted, setMounted] = useState(false);

  // Query form state
  const [queryExpr, setQueryExpr] = useState("");
  const [queryLimit, setQueryLimit] = useState(100);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (mounted) {
      loadCollections();
      loadServerStatus();
    }
  }, [mounted]);

  useEffect(() => {
    if (selectedCollection && mounted) {
      loadCollectionDetails(selectedCollection);
    }
  }, [selectedCollection, mounted]);

  async function loadCollections() {
    if (typeof window === 'undefined') return;
    
    try {
      const API_URL = getApiUrl();
      const response = await fetch(`${API_URL}/api/milvus/collections`);
      if (!response.ok) throw new Error("Failed to load collections");
      const data = await response.json();
      setCollections(data.map((name: string) => ({ name })));
    } catch (err) {
      console.error("Error loading collections:", err);
      setMessage("❌ Failed to load collections. Is the backend running?");
    }
  }

  async function loadServerStatus() {
    if (typeof window === 'undefined') return;
    
    try {
      const API_URL = getApiUrl();
      const response = await fetch(`${API_URL}/api/milvus/server/status`);
      const data = await response.json();
      setServerStatus(data);
    } catch (err) {
      console.error("Error loading server status:", err);
      setServerStatus({
        status: "disconnected",
        host: "unknown",
        port: "unknown",
        collections_count: 0,
        total_entities: 0,
        error: "Cannot connect to backend"
      });
    }
  }

  async function loadCollectionDetails(collectionName: string) {
    if (typeof window === 'undefined') return;
    
    setLoading(true);
    try {
      const API_URL = getApiUrl();
      const response = await fetch(`${API_URL}/api/milvus/collections/${collectionName}`);
      if (!response.ok) throw new Error("Failed to load collection details");
      const data = await response.json();
      setCollectionDetails(data);
      setMessage("");
    } catch (err) {
      console.error("Error loading collection details:", err);
      setMessage("❌ Failed to load collection details");
    } finally {
      setLoading(false);
    }
  }

  async function queryCollection() {
    if (!selectedCollection || typeof window === 'undefined') return;

    setLoading(true);
    try {
      const API_URL = getApiUrl();
      const response = await fetch(`${API_URL}/api/milvus/collections/${selectedCollection}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          collection_name: selectedCollection,
          expr: queryExpr || undefined,
          limit: queryLimit,
          output_fields: ["*"]
        })
      });

      if (!response.ok) throw new Error("Query failed");
      const data = await response.json();
      setQueryResults(data);
      setMessage(`✅ Found ${data.total_results} results`);
    } catch (err) {
      console.error("Query error:", err);
      setMessage("❌ Query failed");
    } finally {
      setLoading(false);
    }
  }

  if (!mounted) {
    return (
      <main className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin text-4xl mb-4">⚙️</div>
          <p>Loading Milvus Dashboard...</p>
        </div>
      </main>
    );
  }

  return (
    <main className={`min-h-screen transition-colors duration-300 ${
      isDark ? "bg-neutral-950 text-white" : "bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 text-gray-900"
    }`}>
      {/* Header */}
      <header className={`border-b backdrop-blur sticky top-0 z-50 ${
        isDark ? "border-neutral-800 bg-neutral-900/80" : "border-gray-200 bg-white/80"
      }`}>
        <div className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          <div>
            <h1 className={`text-2xl font-bold ${isDark ? "text-cyan-400" : "text-blue-600"}`}>
              Milvus Admin Dashboard
            </h1>
            <p className={`text-xs ${isDark ? "text-gray-500" : "text-gray-600"} mt-1`}>
              Collection Management & Data Explorer
            </p>
          </div>
          <div className="flex gap-3 items-center">
            <ThemeToggle />
            <Link href="/" className={`text-sm ${isDark ? "text-gray-400 hover:text-cyan-400" : "text-gray-600 hover:text-blue-600"}`}>
              ← Back to Home
            </Link>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Server Status Card */}
        {serverStatus && (
          <div className={`mb-6 p-6 rounded-lg border-2 ${
            serverStatus.status === "connected"
              ? isDark ? "bg-emerald-950/40 border-emerald-600" : "bg-emerald-50 border-emerald-400"
              : isDark ? "bg-red-950/40 border-red-600" : "bg-red-50 border-red-400"
          }`}>
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold mb-2">Milvus Server Status</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className={isDark ? "text-gray-400" : "text-gray-600"}>Status:</span>
                    <span className={`ml-2 font-bold ${serverStatus.status === "connected" ? "text-emerald-400" : "text-red-400"}`}>
                      {serverStatus.status === "connected" ? "● Connected" : "● Disconnected"}
                    </span>
                  </div>
                  <div>
                    <span className={isDark ? "text-gray-400" : "text-gray-600"}>Host:</span>
                    <span className="ml-2 font-mono">{serverStatus.host}:{serverStatus.port}</span>
                  </div>
                  <div>
                    <span className={isDark ? "text-gray-400" : "text-gray-600"}>Collections:</span>
                    <span className="ml-2 font-bold">{serverStatus.collections_count}</span>
                  </div>
                  <div>
                    <span className={isDark ? "text-gray-400" : "text-gray-600"}>Total Entities:</span>
                    <span className="ml-2 font-bold">{serverStatus.total_entities.toLocaleString()}</span>
                  </div>
                </div>
              </div>
              <button
                onClick={() => { loadCollections(); loadServerStatus(); }}
                className={`px-4 py-2 rounded ${isDark ? "bg-cyan-600 hover:bg-cyan-500" : "bg-blue-600 hover:bg-blue-500"} text-white`}
              >
                🔄 Refresh
              </button>
            </div>
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-2 mb-6">
          {(["collections", "explorer", "stats"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-3 rounded-lg font-semibold capitalize transition-all ${
                activeTab === tab
                  ? isDark
                    ? "bg-cyan-600 text-white"
                    : "bg-blue-600 text-white"
                  : isDark
                  ? "bg-neutral-800 text-gray-400 hover:bg-neutral-700"
                  : "bg-white text-gray-600 hover:bg-gray-100"
              }`}
            >
              {tab === "collections" && "📚"} {tab === "explorer" && "🔍"} {tab === "stats" && "📊"} {tab}
            </button>
          ))}
        </div>

        {/* Message */}
        {message && (
          <div className={`mb-4 p-4 rounded-lg ${
            message.startsWith("✅")
              ? isDark ? "bg-emerald-900/40 text-emerald-200" : "bg-emerald-100 text-emerald-800"
              : isDark ? "bg-red-900/40 text-red-200" : "bg-red-100 text-red-800"
          }`}>
            {message}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Collections List */}
          <div className={`lg:col-span-1 rounded-lg border p-6 ${
            isDark ? "bg-neutral-900 border-neutral-800" : "bg-white border-gray-200"
          }`}>
            <h3 className="text-lg font-bold mb-4">Collections ({collections.length})</h3>
            <div className="space-y-2 max-h-[600px] overflow-y-auto">
              {collections.map((collection) => (
                <button
                  key={collection.name}
                  onClick={() => setSelectedCollection(collection.name)}
                  className={`w-full text-left px-4 py-3 rounded-lg transition-all ${
                    selectedCollection === collection.name
                      ? isDark
                        ? "bg-cyan-900/40 border-2 border-cyan-500"
                        : "bg-blue-100 border-2 border-blue-500"
                      : isDark
                      ? "bg-neutral-800 hover:bg-neutral-700 border-2 border-transparent"
                      : "bg-gray-50 hover:bg-gray-100 border-2 border-transparent"
                  }`}
                >
                  <div className="font-semibold">{collection.name}</div>
                  {collection.num_entities !== undefined && (
                    <div className="text-xs text-gray-500 mt-1">
                      {collection.num_entities.toLocaleString()} entities
                    </div>
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* Main Content Area */}
          <div className={`lg:col-span-2 rounded-lg border p-6 ${
            isDark ? "bg-neutral-900 border-neutral-800" : "bg-white border-gray-200"
          }`}>
            {!selectedCollection ? (
              <div className="text-center py-16">
                <div className="text-5xl mb-4">📚</div>
                <h3 className="text-xl font-semibold mb-2">Select a Collection</h3>
                <p className={isDark ? "text-gray-500" : "text-gray-600"}>
                  Choose a collection from the list to view details
                </p>
              </div>
            ) : loading ? (
              <div className="text-center py-16">
                <div className="animate-spin text-4xl mb-4">⚙️</div>
                <p>Loading...</p>
              </div>
            ) : (
              <>
                {/* Collections Tab */}
                {activeTab === "collections" && collectionDetails && (
                  <div>
                    <h2 className="text-2xl font-bold mb-4">{collectionDetails.name}</h2>
                    <div className="space-y-6">
                      {/* Basic Info */}
                      <div>
                        <h3 className="text-lg font-semibold mb-2">Basic Information</h3>
                        <div className="grid grid-cols-2 gap-4">
                          <div className={`p-4 rounded ${isDark ? "bg-neutral-800" : "bg-gray-50"}`}>
                            <div className={`text-sm ${isDark ? "text-gray-400" : "text-gray-600"}`}>Entities</div>
                            <div className="text-2xl font-bold">{collectionDetails.num_entities.toLocaleString()}</div>
                          </div>
                          <div className={`p-4 rounded ${isDark ? "bg-neutral-800" : "bg-gray-50"}`}>
                            <div className={`text-sm ${isDark ? "text-gray-400" : "text-gray-600"}`}>Fields</div>
                            <div className="text-2xl font-bold">{collectionDetails.schema.fields.length}</div>
                          </div>
                        </div>
                      </div>

                      {/* Schema */}
                      <div>
                        <h3 className="text-lg font-semibold mb-2">Schema</h3>
                        <div className="overflow-x-auto">
                          <table className="w-full text-sm">
                            <thead className={isDark ? "bg-neutral-800" : "bg-gray-100"}>
                              <tr>
                                <th className="px-4 py-2 text-left">Field Name</th>
                                <th className="px-4 py-2 text-left">Type</th>
                                <th className="px-4 py-2 text-left">Primary</th>
                                <th className="px-4 py-2 text-left">Dimension</th>
                              </tr>
                            </thead>
                            <tbody>
                              {collectionDetails.schema.fields.map((field, idx) => (
                                <tr key={idx} className={isDark ? "border-b border-neutral-800" : "border-b border-gray-200"}>
                                  <td className="px-4 py-2 font-mono">{field.name}</td>
                                  <td className="px-4 py-2">
                                    <span className={`px-2 py-1 rounded text-xs ${
                                      isDark ? "bg-cyan-900/40 text-cyan-300" : "bg-cyan-100 text-cyan-800"
                                    }`}>
                                      {field.type}
                                    </span>
                                  </td>
                                  <td className="px-4 py-2">
                                    {field.is_primary ? (
                                      <span className={isDark ? "text-emerald-400" : "text-emerald-600"}>✓</span>
                                    ) : (
                                      <span className={isDark ? "text-gray-600" : "text-gray-400"}>-</span>
                                    )}
                                  </td>
                                  <td className="px-4 py-2">{field.dim || field.max_length || "-"}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>

                      {/* Indexes */}
                      {collectionDetails.indexes.length > 0 && (
                        <div>
                          <h3 className="text-lg font-semibold mb-2">Indexes</h3>
                          {collectionDetails.indexes.map((index, idx) => (
                            <div key={idx} className={`p-4 rounded mb-2 ${isDark ? "bg-neutral-800" : "bg-gray-50"}`}>
                              <div className="font-semibold">{index.field}</div>
                              <div className="text-sm mt-2 space-y-1">
                                <div><span className={isDark ? "text-gray-400" : "text-gray-600"}>Type:</span> {index.index_type}</div>
                                <div><span className={isDark ? "text-gray-400" : "text-gray-600"}>Metric:</span> {index.metric_type}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Explorer Tab */}
                {activeTab === "explorer" && (
                  <div>
                    <h2 className="text-2xl font-bold mb-4">Data Explorer</h2>
                    
                    {/* Query Form */}
                    <div className="mb-6 space-y-4">
                      <div>
                        <label className="block text-sm font-semibold mb-2">Filter Expression</label>
                        <input
                          type="text"
                          value={queryExpr}
                          onChange={(e) => setQueryExpr(e.target.value)}
                          placeholder='e.g., id > 100 or leave empty for all'
                          className={`w-full px-4 py-2 rounded ${
                            isDark
                              ? "bg-neutral-800 border-neutral-700 text-white"
                              : "bg-white border-gray-300 text-gray-900"
                          } border focus:outline-none focus:ring-2 focus:ring-cyan-500`}
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-semibold mb-2">Limit</label>
                          <input
                            type="number"
                            value={queryLimit}
                            onChange={(e) => setQueryLimit(parseInt(e.target.value))}
                            className={`w-full px-4 py-2 rounded ${
                              isDark
                                ? "bg-neutral-800 border-neutral-700 text-white"
                                : "bg-white border-gray-300 text-gray-900"
                            } border focus:outline-none focus:ring-2 focus:ring-cyan-500`}
                          />
                        </div>
                        <div className="flex items-end">
                          <button
                            onClick={queryCollection}
                            disabled={loading}
                            className={`w-full px-6 py-2 rounded font-semibold ${
                              isDark
                                ? "bg-cyan-600 hover:bg-cyan-500 disabled:bg-neutral-700"
                                : "bg-blue-600 hover:bg-blue-500 disabled:bg-gray-400"
                            } text-white transition-colors`}
                          >
                            {loading ? "Querying..." : "🔍 Query"}
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* Query Results */}
                    {queryResults && (
                      <div>
                        <h3 className="text-lg font-semibold mb-2">
                          Results ({queryResults.total_results} found)
                        </h3>
                        
                        {/* Enhanced Results Display */}
                        <div className="space-y-4 max-h-[600px] overflow-auto">
                          {queryResults.results.map((result: any, idx: number) => (
                            <div 
                              key={idx} 
                              className={`p-4 rounded border ${
                                isDark ? "bg-neutral-800 border-neutral-700" : "bg-gray-50 border-gray-200"
                              }`}
                            >
                              <div className="flex justify-between items-start mb-2">
                                <h4 className="font-semibold text-sm">
                                  Document #{idx + 1}
                                </h4>
                                <button
                                  onClick={() => {
                                    const elem = document.getElementById(`result-${idx}`);
                                    if (elem) {
                                      elem.classList.toggle('hidden');
                                    }
                                  }}
                                  className={`text-xs px-2 py-1 rounded ${
                                    isDark ? "bg-cyan-600 hover:bg-cyan-500" : "bg-blue-600 hover:bg-blue-500"
                                  } text-white`}
                                >
                                  Toggle Details
                                </button>
                              </div>
                              
                              <div id={`result-${idx}`}>
                                <div className="space-y-2 text-sm">
                                  {Object.entries(result).map(([key, value]: [string, any]) => (
                                    <div key={key} className="border-b border-neutral-700 pb-2">
                                      <span className={`font-semibold ${isDark ? "text-cyan-400" : "text-blue-600"}`}>
                                        {key}:
                                      </span>
                                      
                                      {/* Check if this is an embedding field */}
                                      {value && typeof value === 'object' && value.type === 'embedding' ? (
                                        <div className="mt-1">
                                          <div className={`text-xs ${isDark ? "text-gray-400" : "text-gray-600"} mb-1`}>
                                            Vector Embedding ({value.dimension} dimensions)
                                          </div>
                                          
                                          {/* Preview */}
                                          <div className={`p-2 rounded text-xs font-mono ${
                                            isDark ? "bg-neutral-900" : "bg-gray-100"
                                          }`}>
                                            <div className="mb-1 font-semibold">Preview (first 5 values):</div>
                                            <div>[{value.preview.map((v: number) => v.toFixed(4)).join(', ')}...]</div>
                                          </div>
                                          
                                          {/* Expandable full embedding */}
                                          <details className="mt-2">
                                            <summary className={`cursor-pointer text-xs ${
                                              isDark ? "text-cyan-400 hover:text-cyan-300" : "text-blue-600 hover:text-blue-500"
                                            }`}>
                                              Show full embedding vector
                                            </summary>
                                            <div className={`mt-2 p-2 rounded text-xs font-mono max-h-40 overflow-auto ${
                                              isDark ? "bg-neutral-900" : "bg-gray-100"
                                            }`}>
                                              [{value.full.map((v: number, i: number) => (
                                                <span key={i}>
                                                  {v.toFixed(6)}
                                                  {i < value.full.length - 1 ? ', ' : ''}
                                                  {(i + 1) % 5 === 0 ? '\n' : ''}
                                                </span>
                                              ))}]
                                            </div>
                                          </details>
                                        </div>
                                      ) : (
                                        /* Regular field display */
                                        <span className="ml-2 break-words">
                                          {typeof value === 'object' 
                                            ? JSON.stringify(value, null, 2) 
                                            : String(value)}
                                        </span>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                        
                        {/* Raw JSON Toggle */}
                        <details className="mt-4">
                          <summary className={`cursor-pointer text-sm font-semibold ${
                            isDark ? "text-cyan-400 hover:text-cyan-300" : "text-blue-600 hover:text-blue-500"
                          }`}>
                            View Raw JSON
                          </summary>
                          <pre className={`mt-2 p-4 rounded text-xs ${
                            isDark ? "bg-neutral-800" : "bg-gray-50"
                          } overflow-x-auto max-h-96`}>
                            {JSON.stringify(queryResults, null, 2)}
                          </pre>
                        </details>
                      </div>
                    )}
                  </div>
                )}

                {/* Stats Tab */}
                {activeTab === "stats" && collectionDetails && (
                  <div>
                    <h2 className="text-2xl font-bold mb-4">Statistics</h2>
                    <div className="grid grid-cols-2 gap-4">
                      <div className={`p-6 rounded ${isDark ? "bg-neutral-800" : "bg-gray-50"}`}>
                        <div className={`text-sm mb-2 ${isDark ? "text-gray-400" : "text-gray-600"}`}>
                          Total Entities
                        </div>
                        <div className="text-3xl font-bold">{collectionDetails.num_entities.toLocaleString()}</div>
                      </div>
                      <div className={`p-6 rounded ${isDark ? "bg-neutral-800" : "bg-gray-50"}`}>
                        <div className={`text-sm mb-2 ${isDark ? "text-gray-400" : "text-gray-600"}`}>
                          Total Fields
                        </div>
                        <div className="text-3xl font-bold">{collectionDetails.schema.fields.length}</div>
                      </div>
                      <div className={`p-6 rounded ${isDark ? "bg-neutral-800" : "bg-gray-50"}`}>
                        <div className={`text-sm mb-2 ${isDark ? "text-gray-400" : "text-gray-600"}`}>
                          Vector Fields
                        </div>
                        <div className="text-3xl font-bold">
                          {collectionDetails.schema.fields.filter(f => f.dim).length}
                        </div>
                      </div>
                      <div className={`p-6 rounded ${isDark ? "bg-neutral-800" : "bg-gray-50"}`}>
                        <div className={`text-sm mb-2 ${isDark ? "text-gray-400" : "text-gray-600"}`}>
                          Indexed Fields
                        </div>
                        <div className="text-3xl font-bold">{collectionDetails.indexes.length}</div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}