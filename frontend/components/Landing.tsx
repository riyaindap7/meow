"use client";

import { useSession, signOut } from "@/lib/auth-client";
import { useTheme } from "@/lib/ThemeContext";
import Link from "next/link";
import { useState } from "react";
import AuthModal from "./AuthModal";

export default function Landing() {
  const { theme } = useTheme();
  const { data: session, isPending } = useSession();
  const [showModal, setShowModal] = useState<'signin' | 'signup' | null>(null);

  const isDark = theme === "dark";

  if (isPending) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500 mx-auto"></div>
          <p className="text-white mt-4">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="min-h-screen bg-gray-900 text-white">
        {/* Navigation */}
        <nav className="flex justify-between items-center p-6 border-b border-gray-700">
          <div className="text-2xl font-bold text-blue-400">⚡ VICTOR</div>
          
          <div className="flex items-center gap-4">
            {session?.user ? (
              <>
                <Link
                  href="/upload"
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition"
                >
                   Upload
                </Link>
                <Link
                  href="/search"
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition"
                >
                   Search
                </Link>
                <Link
                  href="/chat"
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition"
                >
                  💬 Chat
                </Link>
                <div className="flex items-center gap-3">
                  <span className="text-gray-300">
                    Hi, {session.user.name || session.user.email}
                  </span>
                  <button
                    onClick={() => signOut()}
                    className="px-4 py-2 border border-gray-600 hover:bg-gray-700 rounded-lg text-sm transition"
                  >
                    🚪 Sign Out
                  </button>
                </div>
              </>
            ) : (
              <>
                <button
                  onClick={() => setShowModal('signin')}
                  className="px-6 py-2 border border-blue-600 text-blue-400 hover:bg-blue-600/10 rounded-lg font-semibold transition"
                >
                  🔑 Sign In
                </button>
                <button
                  onClick={() => setShowModal('signup')}
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition"
                >
                  📝 Sign Up
                </button>
              </>
            )}
          </div>
        </nav>

        {/* Hero Section */}
        <div className="max-w-7xl mx-auto px-6 py-20">
          <div className="text-center mb-12">
            <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
              AI-Powered Document Intelligence
            </h1>
            <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
              Search and analyze government policies, regulations, and documents with AI-powered RAG technology
            </p>

            {session?.user ? (
              <div className="flex gap-4 justify-center">
                <Link
                  href="/chat"
                  className="px-8 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold text-lg transition"
                >
                  ➡️ Start Chatting
                </Link>
              </div>
            ) : (
              <div className="flex gap-4 justify-center">
                <button
                  onClick={() => setShowModal('signin')}
                  className="px-8 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold text-lg transition"
                >
                  🔑 Sign In Now
                </button>
                <button
                  onClick={() => setShowModal('signup')}
                  className="px-8 py-3 border-2 border-blue-600 text-blue-400 hover:bg-blue-600/10 rounded-lg font-semibold text-lg transition"
                >
                  📝 Create Account
                </button>
              </div>
            )}
          </div>

          {/* Features */}
          <div className="grid md:grid-cols-3 gap-8 mt-20">
            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
              <div className="text-3xl mb-4">🔍</div>
              <h3 className="text-xl font-bold mb-2">Smart Search</h3>
              <p className="text-gray-400">
                Find relevant information across thousands of documents using advanced vector search
              </p>
            </div>

            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
              <div className="text-3xl mb-4">🤖</div>
              <h3 className="text-xl font-bold mb-2">AI Answers</h3>
              <p className="text-gray-400">
                Get accurate answers backed by cited sources from government documents
              </p>
            </div>

            <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
              <div className="text-3xl mb-4">💬</div>
              <h3 className="text-xl font-bold mb-2">Chat History</h3>
              <p className="text-gray-400">
                Maintain context across conversations with persistent chat history
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Auth Modal */}
      <AuthModal
        isOpen={showModal !== null}
        type={showModal || 'signin'}
        onClose={() => setShowModal(null)}
        onSuccess={() => {
          console.log('Auth successful');
          setShowModal(null);
        }}
      />
    </>
  );
}
