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
  const [role, setRole] = useState('user'); // Default role
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
          role, // Include role in signup
        });
      } else {
        await signIn.email({ email, password });
      }

      setEmail('');
      setPassword('');
      setName('');
      setRole('user');
      
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
    if (e.target === e.currentTarget) onClose();
  };

  return (
    <div 
      className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 backdrop-blur-sm"
      onClick={handleOverlayClick}
    >
      <div className="bg-[#0f0f0f] p-8 rounded-2xl max-w-md w-full mx-4 border border-white/20 shadow-xl backdrop-blur">
        
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-semibold text-white">
            {type === 'signup' ? 'Create Account' : 'Welcome Back'}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition text-3xl leading-none"
          >
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          
          {type === 'signup' && (
            <div>
              <label className="text-sm font-medium text-gray-300">Full Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="mt-1 w-full bg-[#1a1a1a] border border-gray-700 rounded-lg p-3 text-white 
                focus:outline-none focus:border-white/40 transition-colors"
                required
                disabled={loading}
              />
            </div>
          )}

          <div>
            <label className="text-sm font-medium text-gray-300">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 w-full bg-[#1a1a1a] border border-gray-700 rounded-lg p-3 text-white
              focus:outline-none focus:border-white/40 transition-colors"
              required
              disabled={loading}
            />
          </div>

          <div>
            <label className="text-sm font-medium text-gray-300">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 w-full bg-[#1a1a1a] border border-gray-700 rounded-lg p-3 text-white 
              focus:outline-none focus:border-white/40 transition-colors"
              required
              disabled={loading}
            />
          </div>

          {type === 'signup' && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Role</label>
              <select
                value={role}
                onChange={(e) => setRole(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg p-3 text-white focus:ring-2 focus:ring-blue-500"
                disabled={loading}
              >
                <option value="user">User</option>
                <option value="research_assistant">Research Assistant</option>
                <option value="policy_maker">Policy Maker</option>
                <option value="admin">Admin</option>
              </select>
            </div>
          )}
          
          {error && (
            <div className="bg-red-900/30 border border-red-500 rounded-lg p-3">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          <div className="flex gap-3 pt-2">
            <button
              type="submit"
              disabled={loading}
              className="flex-1 bg-white hover:bg-gray-300 text-black disabled:opacity-50 disabled:cursor-not-allowed py-3 rounded-lg font-semibold transition"
            >
              {loading ? 'Please wait...' : (type === 'signup' ? 'Sign Up' : 'Sign In')}
            </button>

            <button
              type="button"
              onClick={onClose}
              disabled={loading}
              className="px-6 py-3 border border-white/30 hover:bg-gray-900 disabled:opacity-50 rounded-lg text-gray-300 transition"
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
