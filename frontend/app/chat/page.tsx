'use client';

import { useAuth } from "@/lib/auth-context"; // CHANGED: Use new auth context
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import ChatInterface from '@/components/ChatInterface';

export default function ChatPage() {
  const { user, loading } = useAuth(); // CHANGED: Use new auth context
  const router = useRouter();
  const [authToken, setAuthToken] = useState<string | null>(null);

  useEffect(() => {
    const getAuthToken = async () => {
      if (loading) return;
      
      if (!user) {
        // Redirect to home if not authenticated
        router.push('/');
        return;
      }
      
      try {
        // Use the MongoDB ObjectId as the auth token
        const userId = user._id; // CHANGED: Access _id from user object
        if (userId) {
          setAuthToken(String(userId));
          console.log('Auth token set for user:', user.name || user.email);
        } else {
          console.error('No user ID found in session');
          router.push('/');
        }
      } catch (error) {
        console.error('Failed to get auth token:', error);
        router.push('/');
      }
    };

    getAuthToken();
  }, [user, loading, router]); // CHANGED: Dependencies

  if (loading || !authToken) {
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
    <main className="w-full h-screen">
      <ChatInterface authToken={authToken} />
    </main>
  );
}
