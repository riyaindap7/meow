"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth-context"; // CHANGED: Use new auth context
import { useTheme } from "@/lib/ThemeContext";
import AuthModal from "./AuthModal";
import Image from "next/image";
import Beams from "./Beams";

// ========================= SVG Icon Components =========================
const Icon = {
  Search: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-8 w-8 text-white"
    >
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  ),

  Zap: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-8 w-8 text-white"
    >
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  ),

  ShieldCheck: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-8 w-8 text-white"
    >
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  ),

  Bot: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-8 w-8 mb-4 text-white"
    >
      <path d="M12 8V4H8" />
      <rect x="4" y="12" width="16" height="8" rx="2" />
      <path d="M2 12h2" />
      <path d="M20 12h2" />
      <path d="M12 12v.01" />
    </svg>
  ),
};

// ========================= Landing Page Component =========================
export default function Landing() {
  const { theme } = useTheme();
  const { user, loading, signout } = useAuth(); // CHANGED: Use new auth context
  const [showModal, setShowModal] = useState<"signin" | "signup" | null>(null);
  const [navOpen, setNavOpen] = useState(false);

  if (loading) { // CHANGED: renamed from isPending
    return (
      <div className="flex items-center justify-center min-h-screen bg-black">
        <div className="text-center">
          <div className="animate-spin rounded-full h-24 w-24 border-b-2 border-white/70 mx-auto" />
          <p className="text-white mt-4">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="min-h-screen w-full bg-black text-white overflow-x-hidden transition-colors duration-300">
        {/* Header (pill navbar with auth + nav buttons) */}
        <header className="fixed top-5 left-1/2 -translate-x-1/2 z-50 backdrop-blur-xl border border-white/10 rounded-full shadow-lg bg-white/5 w-[90%] max-w-4xl">
          <div className="px-6 py-3 flex items-center justify-between gap-4">
            {/* Logo / Brand */}
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2">
                <Image
                  src="/logo.png"
                  alt="Victor Logo"
                  width={35}
                  height={35}
                  className="rounded-full"
                />
                <span className="text-xl font-bold text-white tracking-tight">
                  Victor
                </span>
              </div>
            </div>

            {/* Right Side: nav + auth */}
            <div className="flex items-center gap-3">
              {user ? ( // CHANGED: use user directly
                <>
                  {/* Greeting + Hamburger (Sign Out inside menu) */}
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] text-neutral-200 max-w-[140px] truncate">
                      Hi, {user.name || user.email}
                    </span>

                    {/* Hamburger Menu */}
                    <div className="relative">
                      <button
                        onClick={() => setNavOpen((prev) => !prev)}
                        className="p-2 rounded-full border border-white/20 hover:bg-white/10 transition flex items-center justify-center"
                        aria-label="Open navigation menu"
                      >
                        <span className="sr-only">Open navigation</span>
                        <div className="flex flex-col gap-0.5">
                          <span className="w-3.5 h-[1.5px] bg-white rounded-full" />
                          <span className="w-3.5 h-[1.5px] bg-white rounded-full" />
                          <span className="w-3.5 h-[1.5px] bg-white rounded-full" />
                        </div>
                      </button>

                      {navOpen && (
                        <div className="absolute right-0 mt-2 w-40 rounded-2xl bg-black/90 border border-white/15 shadow-xl backdrop-blur-sm py-2 text-sm">
                          <Link
                            href="/upload"
                            onClick={() => setNavOpen(false)}
                            className="block px-4 py-2 hover:bg-white/10"
                          >
                            Upload
                          </Link>
                          <Link
                            href="/search"
                            onClick={() => setNavOpen(false)}
                            className="block px-4 py-2 hover:bg-white/10"
                          >
                            Search
                          </Link>
                          <Link
                            href="/chat"
                            onClick={() => setNavOpen(false)}
                            className="block px-4 py-2 hover:bg-white/10"
                          >
                            Chat
                          </Link>
                          <button
                            onClick={() => {
                              setNavOpen(false);
                              signout();
                            }}
                            className="w-full text-left px-4 py-2 hover:bg-red-500/20 text-red-300"
                          >
                            Sign Out
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </>
              ) : (
                // When logged out: Sign In / Sign Up buttons
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setShowModal("signin")}
                    className="px-3 py-1.5 rounded-full border border-white/30 text-xs font-medium text-neutral-100 hover:bg-white/10 transition"
                  >
                    Sign In
                  </button>
                  <button
                    onClick={() => setShowModal("signup")}
                    className="px-3 py-1.5 rounded-full bg-white text-xs font-semibold text-black hover:bg-neutral-200 transition"
                  >
                    Sign Up
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>

        <main>
          {/* ========================= Hero Section ========================= */}
          <section className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 text-center bg-black pt-32">
            {/* Background Beams */}
            <div className="absolute inset-0 -z-10 opacity-80 h-full">
              <Beams
                beamWidth={2}
                beamHeight={50}
                beamNumber={12}
                lightColor="#ffffff"
                speed={2}
                noiseIntensity={1.75}
                scale={0.2}
                rotation={30}
              />
            </div>

            <h1 className="relative text-6xl md:text-7xl lg:text-8xl font-black uppercase leading-tight mb-4 text-white">
              Navigate Complexity
            </h1>

            <h2 className="relative text-4xl md:text-5xl lg:text-6xl font-bold uppercase mb-8 text-white">
              Efficient. Fast Retrieval.
            </h2>

            <p className="relative max-w-3xl text-lg mb-12 md:mb-16 leading-relaxed text-neutral-300">
              Transform vast regulatory databases into an intelligent, queryable
              knowledge base. Get instant answers backed by verified sources.
            </p>

            {/* Hero CTAs: session-aware */}
            <div className="relative flex gap-4 flex-wrap justify-center">
              {user ? ( // CHANGED
                <>
                  <Link
                    href="/chat"
                    className="inline-flex items-center justify-center px-10 py-4 font-mono font-medium tracking-tighter text-black bg-white hover:bg-neutral-200 rounded-lg transition-all transform hover:scale-105 shadow-lg"
                  >
                    Start Chatting
                  </Link>
                  <Link
                    href="/search"
                    className="inline-flex items-center justify-center px-10 py-4 font-mono font-medium tracking-tighter text-white bg-gradient-to-r from-neutral-700 to-neutral-400 hover:from-neutral-600 hover:to-neutral-300 rounded-lg transition-all transform hover:scale-105 shadow-lg"
                  >
                    Search Documents
                  </Link>
                  <Link
                    href="/upload"
                    className="inline-flex items-center justify-center px-10 py-4 font-mono font-medium tracking-tighter text-white bg-neutral-900 hover:bg-neutral-800 rounded-lg transition-all border border-neutral-600 transform hover:scale-105"
                  >
                    Upload PDFs
                  </Link>
                </>
              ) : (
                <>
                  <button
                    onClick={() => setShowModal("signin")}
                    className="inline-flex items-center justify-center px-10 py-4 font-mono font-medium tracking-tighter text-black bg-white hover:bg-neutral-200 rounded-lg transition-all transform hover:scale-105 shadow-lg"
                  >
                    Sign In Now
                  </button>
                  <button
                    onClick={() => setShowModal("signup")}
                    className="inline-flex items-center justify-center px-10 py-4 font-mono font-medium tracking-tighter text-white bg-black border border-white/40 hover:bg-white/5 rounded-lg transition-all transform hover:scale-105"
                  >
                    Create Account
                  </button>
                </>
              )}
            </div>
          </section>

          {/* ========================= About Section ========================= */}
          <section id="about" className="py-20 px-4 relative z-10 bg-black">
            <div className="max-w-6xl mx-auto">
              <h2 className="text-4xl font-bold mb-8 text-center text-white">
                About Victor
              </h2>
              <div className="max-w-4xl mx-auto bg-neutral-950 border border-neutral-800 rounded-lg p-8 hover:border-neutral-600 transition-all hover:shadow-lg hover:shadow-black/40">
                <p className="text-neutral-200 leading-relaxed text-lg">
                  Victor is an advanced AI-powered information retrieval tool
                  designed specifically for navigating the complex web of
                  government regulations, policies, and schemes. Our mission is
                  to empower decision-makers by transforming vast, unstructured
                  databases into an intelligent, queryable knowledge base. By
                  leveraging state-of-the-art technologies, Victor provides
                  instant, accurate, and verifiable answers, fostering a new era
                  of efficiency and data-driven governance.
                </p>
              </div>
            </div>
          </section>

          {/* ========================= Features ========================= */}
          <section id="features" className="py-20 px-4 bg-black">
            <div className="max-w-6xl mx-auto">
              <h2 className="text-4xl font-bold mb-4 text-center text-white">
                The Future of Information Retrieval
              </h2>
              <p className="text-neutral-400 max-w-3xl mx-auto mb-16 text-center">
                Victor transforms how you interact with vast regulatory
                databases, providing unparalleled speed, accuracy, and trust.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
                {[
                  {
                    icon: <Icon.Zap />,
                    title: "Instant Retrieval",
                    desc: "Ask questions in plain English and get answers in seconds, not hours.",
                  },
                  {
                    icon: <Icon.ShieldCheck />,
                    title: "Verifiable Sources",
                    desc: "Every answer is backed by direct citations to the original source document.",
                  },
                  {
                    icon: <Icon.Search />,
                    title: "Hybrid Search",
                    desc: "Combines semantic and keyword search for unparalleled accuracy.",
                  },
                  {
                    icon: <Icon.Bot />,
                    title: "Advanced AI Models",
                    desc: "Utilizes state-of-the-art LLMs to synthesize clear, concise answers.",
                  },
                ].map(({ icon, title, desc }) => (
                  <div
                    key={title}
                    className="feature-card flex flex-col items-center text-center p-8 rounded-lg border border-neutral-800 hover:border-neutral-500 transition-all duration-300 bg-neutral-950"
                  >
                    <div className="mb-4">{icon}</div>
                    <h3 className="text-xl font-semibold mb-2 text-white">
                      {title}
                    </h3>
                    <p className="text-neutral-300 text-sm">{desc}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* ========================= Metrics ========================= */}
          <section className="py-20 px-4 bg-black relative z-10">
            <div className="max-w-6xl mx-auto">
              <h2 className="text-4xl font-bold mb-12 text-center text-white">
                Prototype Testing Results
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                {[
                  { value: "30+", label: "Documents Indexed" },
                  { value: "99%", label: "Retrieval Accuracy" },
                  { value: "95%", label: "Reduction in Research Time" },
                ].map(({ value, label }) => (
                  <div
                    key={label}
                    className="p-8 bg-neutral-950 border border-neutral-800 rounded-lg hover:border-neutral-500 transition-all hover:shadow-lg hover:shadow-black/40 text-center group"
                  >
                    <h3 className="text-5xl font-bold text-white mb-2">
                      {value}
                    </h3>
                    <p className="text-neutral-300">{label}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* ========================= Upload Section ========================= */}
          <section className="py-20 px-4 border-t border-neutral-800 bg-black">
            <div className="max-w-6xl mx-auto">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
                <div>
                  <h2 className="text-4xl font-bold mb-6 text-white">
                    Upload Your Documents
                  </h2>
                  <p className="text-neutral-300 text-lg mb-8">
                    Start by uploading your PDF documents. Victor will
                    automatically process and index them, making them instantly
                    searchable with AI-powered queries.
                  </p>
                  <div className="space-y-4">
                    {[
                      {
                        title: "Multiple Formats",
                        desc: "Support for PDF and document formats",
                      },
                      {
                        title: "Instant Indexing",
                        desc: "Fast processing with vector embeddings",
                      },
                      {
                        title: "Secure Storage",
                        desc: "Your documents are encrypted and safe",
                      },
                    ].map(({ title, desc }) => (
                      <div className="flex gap-3" key={title}>
                        <span className="text-white text-2xl">•</span>
                        <div>
                          <h4 className="text-lg font-semibold mb-1 text-white">
                            {title}
                          </h4>
                          <p className="text-neutral-400 text-sm">{desc}</p>
                        </div>
                      </div>
                    ))}
                  </div>

                  {user ? ( // CHANGED
                    <Link
                      href="/upload"
                      className="inline-flex items-center justify-center mt-8 px-10 py-4 bg-white text-black hover:bg-neutral-200 rounded-lg font-medium transition-colors"
                    >
                      Upload Documents Now
                    </Link>
                  ) : (
                    <button
                      onClick={() => setShowModal("signin")}
                      className="inline-flex items-center justify-center mt-8 px-10 py-4 bg-white text-black hover:bg-neutral-200 rounded-lg font-medium transition-colors"
                    >
                      Sign In to Upload
                    </button>
                  )}
                </div>

                <div className="bg-neutral-950 border border-neutral-800 rounded-lg p-12 text-center">
                  <div className="text-4xl mb-4 text-white">📄</div>
                  <h3 className="text-2xl font-bold mb-4 text-white">
                    Upload & Index
                  </h3>
                  <p className="text-neutral-300 mb-8">
                    Drag and drop your documents or click to browse
                  </p>
                  <div className="bg-black rounded border-2 border-dashed border-neutral-600 p-8">
                    <p className="text-neutral-400">
                      PDF files, Word documents, and more
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* ========================= CTA Section ========================= */}
          <section className="py-20 px-4 border-t border-neutral-800 bg-black">
            <div className="max-w-6xl mx-auto text-center">
              <h2 className="text-4xl font-bold mb-6 text-white">
                Ready to Transform Your Research?
              </h2>
              <p className="text-neutral-300 mb-8 max-w-2xl mx-auto">
                Upload your documents and start searching with Victor&apos;s
                AI-powered retrieval system today.
              </p>
              <div className="flex gap-4 justify-center">
                {user ? ( // CHANGED
                  <>
                    <Link
                      href="/upload"
                      className="inline-flex items-center justify-center px-10 py-4 bg-white text-black hover:bg-neutral-200 rounded-lg font-medium transition-colors"
                    >
                      Upload Documents
                    </Link>
                    <Link
                      href="/search"
                      className="inline-flex items-center justify-center px-10 py-4 bg-neutral-900 hover:bg-neutral-800 text-white rounded-lg font-medium transition-colors border border-neutral-700"
                    >
                      Search Now
                    </Link>
                  </>
                ) : (
                  <>
                    <button
                      onClick={() => setShowModal("signup")}
                      className="inline-flex items-center justify-center px-10 py-4 bg-white text-black hover:bg-neutral-200 rounded-lg font-medium transition-colors"
                    >
                      Create Account
                    </button>
                    <button
                      onClick={() => setShowModal("signin")}
                      className="inline-flex items-center justify-center px-10 py-4 bg-neutral-900 hover:bg-neutral-800 text-white rounded-lg font-medium transition-colors border border-neutral-700"
                    >
                      Sign In
                    </button>
                  </>
                )}
              </div>
            </div>
          </section>
        </main>

        {/* ========================= Footer ========================= */}
        <footer className="border-t border-neutral-800 bg-black mt-20">
          <div className="max-w-6xl mx-auto px-6 py-12">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
              <div>
                <h3 className="text-xl font-bold text-white">Victor</h3>
                <p className="text-neutral-400 mt-2 text-sm">
                  AI-Powered Information Retrieval
                </p>
              </div>

              {[
                {
                  title: "Product",
                  links: ["Features", "Search", "Documentation"],
                },
                {
                  title: "Company",
                  links: ["About Us", "Contact"],
                },
                {
                  title: "Legal",
                  links: ["Privacy Policy", "Terms of Service"],
                },
              ].map(({ title, links }) => (
                <div key={title}>
                  <h4 className="font-semibold mb-4 text-white">{title}</h4>
                  <ul className="space-y-2">
                    {links.map((link) => (
                      <li key={link}>
                        <a
                          href="#"
                          className="text-neutral-400 hover:text-white text-sm transition-colors"
                        >
                          {link}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>

            <div className="text-center py-6 text-neutral-500 text-sm border-t border-neutral-800">
              © 2025 Victor. All Rights Reserved. | Powered by Milvus &
              OpenRouter
            </div>
          </div>
        </footer>

        {/* ========================= Styles ========================= */}
        <style jsx global>{`
          @import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap");

          html {
            scroll-behavior: smooth;
          }

          body {
            margin: 0;
            padding: 0;
            overflow-x: hidden;
            font-family: "Inter", sans-serif;
          }

          .feature-card {
            background: linear-gradient(
              145deg,
              rgba(255, 255, 255, 0.02),
              rgba(255, 255, 255, 0)
            );
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
          }

          .feature-card:before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(
              circle at 50% 0%,
              rgba(255, 255, 255, 0.1),
              transparent 70%
            );
            opacity: 0;
            transition: opacity 0.3s ease;
            transform: translateY(100%);
          }

          .feature-card:hover {
            transform: translateY(-5px);
            border-color: rgba(255, 255, 255, 0.3) !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
          }

          .feature-card:hover:before {
            opacity: 1;
            transform: translateY(0);
          }
        `}</style>
      </div>

      {/* Auth Modal */}
      {showModal !== null && (
        <AuthModal
          mode={showModal} // CHANGED: renamed from type to mode
          onClose={() => setShowModal(null)}
        />
      )}
    </>
  );
}
