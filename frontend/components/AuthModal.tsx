'use client';

import { useState } from 'react';
import { signIn, signUp } from "@/lib/auth-client";

interface AuthModalProps {
  isOpen: boolean;
  type: 'signin' | 'signup';
  onClose: () => void;
  onSuccess?: () => void;
}

export default function AuthModal({ isOpen, type, onClose, onSuccess }: AuthModalProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      if (type === 'signup') {
        await signUp.email({
          email,
          password,
          name,
        });
      } else {
        await signIn.email({
          email,
          password,
        });
      }
      
      // Reset form
      setEmail('');
      setPassword('');
      setName('');
      
      if (onSuccess) onSuccess();
      onClose();
    } catch (err: any) {
      console.error('Auth error:', err);
      setError(err.message || `${type} failed`);
    } finally {
      setLoading(false);
    }
  };

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div 
      className="fixed inset-0 bg-black/70 flex items-center justify-center z-50"
      onClick={handleOverlayClick}
    >
      <div className="bg-gray-800 p-8 rounded-lg max-w-md w-full mx-4 border border-gray-700">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-white">
            {type === 'signup' ? 'Create Account' : 'Sign In'}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl"
          >
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {type === 'signup' && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500"
                required
                disabled={loading}
              />
            </div>
          )}
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500"
              required
              disabled={loading}
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500"
              required
              disabled={loading}
            />
          </div>
          
          {error && (
            <div className="bg-red-900/20 border border-red-500 rounded-lg p-3">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}
          
          <div className="flex gap-3">
            <button
              type="submit"
              disabled={loading}
              className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed py-3 rounded-lg font-semibold text-white transition"
            >
              {loading ? 'Loading...' : (type === 'signup' ? 'Create Account' : 'Sign In')}
            </button>
            <button
              type="button"
              onClick={onClose}
              disabled={loading}
              className="px-6 py-3 border border-gray-600 hover:bg-gray-700 disabled:opacity-50 rounded-lg text-gray-300 transition"
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}