"use client";

import { useAuth } from "@/lib/auth-context";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import ChatInterface from "@/components/ChatInterface";

export default function ChatPage() {
  const { user, token, loading } = useAuth(); // Get token from context
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) {
      router.push("/");
    }
  }, [user, loading, router]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500 mx-auto"></div>
          <p className="text-white mt-4">Loading...</p>
        </div>
      </div>
    );
  }

  if (!user || !token) {
    return null;
  }

  return (
    <main className="w-full h-screen">
      <ChatInterface authToken={token} userName={user.name} /> {/* Use token, not user._id */}
    </main>
  );
}
