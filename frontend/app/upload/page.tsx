"use client";

import UploadButton from "@/components/UploadButton";
import Link from "next/link";

export default function Upload() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      {/* Navigation */}
      <nav className="flex justify-between items-center px-8 py-6 bg-white bg-opacity-80 backdrop-blur-md shadow-sm">
        <Link href="/" className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">
          Victor
        </Link>
        <Link href="/" className="text-gray-600 hover:text-blue-600 transition font-semibold">
          ← Back to Home
        </Link>
      </nav>

      {/* Main Content */}
      <section className="max-w-2xl mx-auto px-8 py-24">
        <div className="space-y-8">
          <div className="text-center space-y-4">
            <h1 className="text-5xl font-bold text-gray-900">
              Upload Your Document
            </h1>
            <p className="text-xl text-gray-600">
              Let Victor analyze and process your files with AI-powered intelligence
            </p>
          </div>

          {/* Upload Card */}
          <div className="bg-white rounded-2xl shadow-xl p-12 space-y-8">
            <div className="border-2 border-dashed border-blue-300 rounded-xl p-8 text-center hover:border-blue-500 transition">
              <UploadButton />
            </div>

            {/* File Info */}
            <div className="bg-blue-50 rounded-lg p-6 space-y-3">
              <h3 className="font-semibold text-gray-900">Supported Formats:</h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-2xl mb-2">📄</div>
                  <p className="text-sm text-gray-600">PDF</p>
                </div>
                <div className="text-center">
                  <div className="text-2xl mb-2">📝</div>
                  <p className="text-sm text-gray-600">DOC/DOCX</p>
                </div>
                <div className="text-center">
                  <div className="text-2xl mb-2">📊</div>
                  <p className="text-sm text-gray-600">TXT</p>
                </div>
              </div>
            </div>

            {/* Features */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-gray-200">
              <div className="text-center">
                <div className="text-3xl mb-2">⚡</div>
                <p className="text-sm text-gray-600">Fast Processing</p>
              </div>
              <div className="text-center">
                <div className="text-3xl mb-2">🔒</div>
                <p className="text-sm text-gray-600">Secure & Private</p>
              </div>
              <div className="text-center">
                <div className="text-3xl mb-2">🤖</div>
                <p className="text-sm text-gray-600">AI-Powered</p>
              </div>
            </div>
          </div>

          {/* Additional Info */}
          <div className="bg-gradient-to-r from-blue-100 to-purple-100 rounded-xl p-6 space-y-3">
            <h3 className="font-semibold text-gray-900">What Happens Next?</h3>
            <ol className="space-y-2 text-gray-700">
              <li className="flex items-start gap-3">
                <span className="font-bold text-blue-600">1.</span>
                <span>Upload your document</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="font-bold text-blue-600">2.</span>
                <span>Victor analyzes the content</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="font-bold text-blue-600">3.</span>
                <span>Get insights and extracted data</span>
              </li>
            </ol>
          </div>
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
