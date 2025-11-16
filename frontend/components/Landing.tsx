import UploadButton from "./UploadButton";
import Link from "next/link";

export default function Landing() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      {/* Navigation */}
      <nav className="flex justify-between items-center px-8 py-6 bg-white bg-opacity-80 backdrop-blur-md shadow-sm">
        <div className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">
          Victor
        </div>
        <div className="space-x-6 flex items-center">
          <a href="#features" className="text-gray-600 hover:text-blue-600 transition">Features</a>
          <Link href="/upload" className="px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:shadow-lg transition">
            Upload
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="max-w-6xl mx-auto px-8 py-24">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-6">
            <h1 className="text-5xl lg:text-6xl font-bold text-gray-900 leading-tight">
              Meet <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">Victor</span>
            </h1>
            <p className="text-xl text-gray-600">
              Your intelligent document processing companion. Upload, analyze, and extract insights from your documents using advanced AI.
            </p>
            <div className="flex gap-4">
              <Link href="/upload" className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:shadow-lg transition transform hover:scale-105 inline-block">
                Start Now
              </Link>
              <button className="px-8 py-3 border-2 border-gray-300 text-gray-700 rounded-lg font-semibold hover:border-blue-600 hover:text-blue-600 transition">
                Learn More
              </button>
            </div>
          </div>
          
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-400 rounded-2xl blur-3xl opacity-20"></div>
            <div className="relative bg-white rounded-2xl shadow-2xl p-8 space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
              </div>
              <div className="space-y-3">
                <div className="h-3 bg-gray-200 rounded w-3/4"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                <div className="h-3 bg-gray-200 rounded w-5/6"></div>
              </div>
              <div className="pt-4 border-t border-gray-200">
                <p className="text-sm text-gray-500">Processing: document.pdf</p>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div className="bg-gradient-to-r from-blue-600 to-purple-600 h-2 rounded-full w-3/4"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="bg-white py-20">
        <div className="max-w-6xl mx-auto px-8">
          <h2 className="text-4xl font-bold text-center mb-16 text-gray-900">Powerful Features</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div className="p-8 rounded-xl border border-gray-200 hover:shadow-lg hover:border-blue-300 transition">
              <div className="text-4xl mb-4">📄</div>
              <h3 className="text-xl font-bold mb-3 text-gray-900">Smart Upload</h3>
              <p className="text-gray-600">Upload PDF, DOC, and DOCX files with instant processing and validation.</p>
            </div>

            {/* Feature 2 */}
            <div className="p-8 rounded-xl border border-gray-200 hover:shadow-lg hover:border-purple-300 transition">
              <div className="text-4xl mb-4">🤖</div>
              <h3 className="text-xl font-bold mb-3 text-gray-900">AI Analysis</h3>
              <p className="text-gray-600">Advanced machine learning models analyze and understand your content.</p>
            </div>

            {/* Feature 3 */}
            <div className="p-8 rounded-xl border border-gray-200 hover:shadow-lg hover:border-pink-300 transition">
              <div className="text-4xl mb-4">⚡</div>
              <h3 className="text-xl font-bold mb-3 text-gray-900">Fast & Reliable</h3>
              <p className="text-gray-600">Lightning-fast processing with enterprise-grade reliability and security.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Upload Section */}
      <section id="upload" className="max-w-6xl mx-auto px-8 py-24">
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-12 text-white text-center space-y-8">
          <h2 className="text-4xl font-bold">Ready to Get Started?</h2>
          <p className="text-lg text-blue-100">Upload your document now and let Victor do the magic.</p>
          <Link href="/upload" className="px-8 py-3 bg-white text-blue-600 rounded-lg font-semibold hover:shadow-lg transition transform hover:scale-105 inline-block">
            Go to Upload
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-gray-400 py-12">
        <div className="max-w-6xl mx-auto px-8 text-center">
          <p>&copy; 2024 Victor. All rights reserved.</p>
          <div className="space-x-4 mt-4">
            <a href="#" className="hover:text-white transition">Privacy</a>
            <a href="#" className="hover:text-white transition">Terms</a>
            <a href="#" className="hover:text-white transition">Contact</a>
          </div>
        </div>
      </footer>
    </main>
  );
}
