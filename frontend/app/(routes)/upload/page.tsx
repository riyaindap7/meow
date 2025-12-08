"use client";

import { useState } from "react";
import Link from "next/link";
import ThemeToggle from "@/components/ThemeToggle";
import { useTheme } from "@/lib/ThemeContext";

export default function Upload() {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");

  // Main Upload Handler - Updated for local backend storage
  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setMessage("");
    console.log("Selected file:", file);

    // Prepare FormData for multipart upload
    const formData = new FormData();
    formData.append("file", file);
    formData.append("org_id", "TEMP_ORG");
    formData.append("uploader_id", "TEMP_USER");
    formData.append("category", "uploads");

    console.log("Sending file to backend /api/upload/upload-direct");

    try {
      const response = await fetch("http://localhost:8000/api/upload/upload-direct", {
        method: "POST",
        body: formData,
        // Don't set Content-Type header - browser will set it with boundary for multipart/form-data
      });

      console.log("Backend response status:", response.status);

      // Parse response
      let respData;
      try {
        respData = await response.json();
      } catch (err) {
        respData = await response.text();
      }

      if (!response.ok) {
        console.error("Backend error:", respData);
        setMessage("Server rejected the upload.");
        setUploading(false);
        return;
      }

      console.log("Backend accepted the file!", respData);
      setMessage("Upload successful! Processing complete.");
    } catch (err) {
      console.error("Network/Fetch error:", err);
      setMessage("Network error contacting backend.");
    } finally {
      setUploading(false);
    }
  }

  return (
  <main className="min-h-screen bg-black text-white transition-colors duration-300">
    
    {/* NAVBAR */}
    <nav className="flex justify-between items-center px-8 py-6 bg-neutral-900 border-b border-neutral-700">
      <Link href="/" className="text-2xl font-semibold text-white hover:text-gray-300 transition">
        Victor
      </Link>

      <div className="flex gap-4 items-center">
        <ThemeToggle />
        <Link
          href="/"
          className="text-gray-400 hover:text-white transition font-medium"
        >
          ← Back
        </Link>
      </div>
    </nav>

    {/* MAIN CONTENT */}
    <section className="max-w-2xl mx-auto px-8 py-24">
      <div className="space-y-4 text-center">
        <h1 className="text-5xl font-bold">Upload Your Document</h1>
        <p className="text-gray-500 text-lg">
          Upload a file and let Victor process it intelligently.
        </p>
      </div>

      {/* Upload Card */}
      <div className="mt-12 rounded-xl p-10 bg-neutral-950 border border-neutral-800 shadow-xl space-y-8">

        {/* Upload Box */}
        <div
          className="border-2 border-dashed border-gray-600 hover:border-white transition rounded-xl p-10 text-center cursor-pointer"
        >
          <input type="file" onChange={handleUpload} className="cursor-pointer" />

          {uploading && <p className="mt-4 text-gray-300">Uploading...</p>}
          {message && <p className="mt-4 text-green-400 font-semibold">{message}</p>}
        </div>

        {/* Supported Formats */}
        <div className="bg-neutral-900 border border-neutral-700 rounded-lg p-6">
          <h3 className="font-semibold text-white">Supported Formats</h3>
          <div className="grid grid-cols-3 gap-4 text-center mt-4 text-gray-400">
            <p>PDF</p>
            <p>DOC/DOCX</p>
            <p>TXT</p>
          </div>
        </div>

        {/* Info Section */}
        <div className="bg-neutral-900 border border-neutral-700 rounded-lg p-6 space-y-3">
          <h3 className="font-semibold text-white">What Happens Next?</h3>
          <ul className="space-y-2 text-gray-400 text-sm">
            <li>• File securely uploaded</li>
            <li>• Server processes content</li>
            <li>• Insights generated automatically</li>
          </ul>
        </div>
      </div>
    </section>

    {/* FOOTER */}
    <footer className="text-gray-600 text-center py-10 border-t border-neutral-800">
      © 2024 Victor. All rights reserved.
    </footer>

  </main>
);

}
