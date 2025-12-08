'use client';

import { useSession, getSession } from "@/lib/auth-client";
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import ChatInterface from '@/components/ChatInterface';

export default function ChatPage() {
  const { data: session, isPending } = useSession();
  const router = useRouter();
  const [authToken, setAuthToken] = useState<string | null>(null);

  useEffect(() => {
    const getAuthToken = async () => {
      if (isPending) return;
      
      if (!session?.user) {
        // Redirect to home if not authenticated
        router.push('/');
        return;
      }
      
      // For Better Auth, use the user ID as token for now
      // In production, you'd get the actual JWT token from cookies or session
      try {
        const userId = session.user.id;
        if (userId) {
          // Use the MongoDB ObjectId as the auth token
          setAuthToken(String(userId));
          console.log('✅ Auth token set for user:', session.user.name || session.user.email);
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
  }, [session, isPending, router]);

  if (isPending || !authToken) {
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
