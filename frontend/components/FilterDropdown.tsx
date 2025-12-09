"use client";

import { useState, useEffect } from "react";
import { Filter, X, Calendar } from "lucide-react";

interface FilterDropdownProps {
  onFilterChange: (filters: SearchFilters) => void;
  currentFilters: SearchFilters;
}

export interface SearchFilters {
  category?: string;
  language?: string;
  document_type?: string;
  document_id?: string;
  date_from?: string;
  date_to?: string;
}

const CATEGORIES = [
  "Scraped_moe_archived_press_releases",
  "Scraped_moe_archived_scholarships",
  "moe_scraped_higher_edu_RUSA",
  "scraped_moe_archived_circulars",
  "scraped_moe_documents&reports",
];

const LANGUAGES = [
  "English",
  "Bilingual",
  "Hindi",
  "Tamil",
  "Telugu",
  "Bengali",
  "Marathi",
];

const DOCUMENT_TYPES = [
  "pdf",
  "PDF",
  "doc",
  "docx",
  "txt",
];

export default function FilterDropdown({ onFilterChange, currentFilters }: FilterDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [filters, setFilters] = useState<SearchFilters>(currentFilters);

  const hasActiveFilters = Object.values(filters).some(val => val && val.trim() !== "");

  const handleFilterChange = (key: keyof SearchFilters, value: string) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
  };

  const applyFilters = () => {
    onFilterChange(filters);
    setIsOpen(false);
  };

  const clearFilters = () => {
    const emptyFilters: SearchFilters = {};
    setFilters(emptyFilters);
    onFilterChange(emptyFilters);
  };

  const clearSingleFilter = (key: keyof SearchFilters) => {
    const newFilters = { ...filters };
    delete newFilters[key];
    setFilters(newFilters);
  };

  return (
    <div className="relative">
      {/* Filter Button - ✅ Added type="button" to prevent form submission */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={`p-3 rounded-lg transition-all flex items-center gap-2 ${
          hasActiveFilters
            ? "bg-blue-500/20 border border-blue-500/50 text-blue-400"
            : "bg-neutral-800/60 border border-neutral-700/50 text-neutral-300 hover:bg-neutral-700/60"
        }`}
        title="Filter documents"
      >
        <Filter className="w-5 h-5" />
        {hasActiveFilters && (
          <span className="text-xs font-semibold bg-blue-500 text-white rounded-full w-5 h-5 flex items-center justify-center">
            {Object.values(filters).filter(v => v && v.trim() !== "").length}
          </span>
        )}
      </button>

      {/* Filter Dropdown - OPENS UPWARD */}
      {isOpen && (
        <div className="absolute bottom-14 left-0 w-96 bg-neutral-900/98 backdrop-blur-xl rounded-xl shadow-2xl border border-neutral-700/50 z-50 max-h-[500px] overflow-hidden flex flex-col">
          {/* Header - Sticky */}
          <div className="bg-neutral-900/98 border-b border-neutral-700/50 px-4 py-3 flex items-center justify-between flex-shrink-0">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-neutral-400" />
              <h3 className="font-semibold text-neutral-200">Filter Documents</h3>
            </div>
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              className="p-1 rounded-lg hover:bg-neutral-800 text-neutral-400 hover:text-neutral-100"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Scrollable Filter Fields */}
          <div className="overflow-y-auto flex-1 p-4 space-y-4">
            {/* Category Filter */}
            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-2">
                Category
              </label>
              <select
                value={filters.category || ""}
                onChange={(e) => handleFilterChange("category", e.target.value)}
                className="w-full px-3 py-2 bg-neutral-800/60 border border-neutral-700/50 rounded-lg text-neutral-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              >
                <option value="">All Categories</option>
                {CATEGORIES.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat.replace(/_/g, " ").replace(/scraped |moe /gi, "")}
                  </option>
                ))}
              </select>
            </div>

            {/* Language Filter */}
            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-2">
                Language
              </label>
              <select
                value={filters.language || ""}
                onChange={(e) => handleFilterChange("language", e.target.value)}
                className="w-full px-3 py-2 bg-neutral-800/60 border border-neutral-700/50 rounded-lg text-neutral-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              >
                <option value="">All Languages</option>
                {LANGUAGES.map((lang) => (
                  <option key={lang} value={lang}>
                    {lang}
                  </option>
                ))}
              </select>
            </div>

            {/* Document Type Filter */}
            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-2">
                Document Type
              </label>
              <select
                value={filters.document_type || ""}
                onChange={(e) => handleFilterChange("document_type", e.target.value)}
                className="w-full px-3 py-2 bg-neutral-800/60 border border-neutral-700/50 rounded-lg text-neutral-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              >
                <option value="">All Types</option>
                {DOCUMENT_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            {/* Document ID Search */}
            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-2">
                Document ID (contains)
              </label>
              <input
                type="text"
                value={filters.document_id || ""}
                onChange={(e) => handleFilterChange("document_id", e.target.value)}
                placeholder="e.g., RTEAct, NEP2020, RUSA"
                className="w-full px-3 py-2 bg-neutral-800/60 border border-neutral-700/50 rounded-lg text-neutral-200 text-sm placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              />
            </div>

            {/* Date Range Filter */}
            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-2 flex items-center gap-2">
                <Calendar className="w-4 h-4" />
                Published Date Range
              </label>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <input
                    type="date"
                    value={filters.date_from || ""}
                    onChange={(e) => handleFilterChange("date_from", e.target.value)}
                    className="w-full px-3 py-2 bg-neutral-800/60 border border-neutral-700/50 rounded-lg text-neutral-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                  />
                  <p className="text-xs text-neutral-500 mt-1">From</p>
                </div>
                <div>
                  <input
                    type="date"
                    value={filters.date_to || ""}
                    onChange={(e) => handleFilterChange("date_to", e.target.value)}
                    className="w-full px-3 py-2 bg-neutral-800/60 border border-neutral-700/50 rounded-lg text-neutral-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                  />
                  <p className="text-xs text-neutral-500 mt-1">To</p>
                </div>
              </div>
            </div>

            {/* Active Filters */}
            {hasActiveFilters && (
              <div>
                <p className="text-xs font-medium text-neutral-400 mb-2">Active Filters:</p>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(filters).map(([key, value]) => {
                    if (!value || value.trim() === "") return null;
                    return (
                      <button
                        key={key}
                        type="button"
                        onClick={() => clearSingleFilter(key as keyof SearchFilters)}
                        className="px-2 py-1 bg-blue-500/20 border border-blue-500/50 text-blue-300 rounded-md text-xs flex items-center gap-1 hover:bg-blue-500/30 transition"
                      >
                        <span className="font-medium">{key}:</span>
                        <span>{value.length > 20 ? value.substring(0, 20) + "..." : value}</span>
                        <X className="w-3 h-3" />
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Action Buttons - Sticky */}
          <div className="bg-neutral-900/98 border-t border-neutral-700/50 px-4 py-3 flex gap-2 flex-shrink-0">
            <button
              type="button"
              onClick={clearFilters}
              disabled={!hasActiveFilters}
              className="flex-1 px-4 py-2 bg-neutral-800/60 border border-neutral-700/50 text-neutral-300 rounded-lg text-sm font-medium hover:bg-neutral-700/60 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              Clear All
            </button>
            <button
              type="button"
              onClick={applyFilters}
              className="flex-1 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg text-sm font-medium transition"
            >
              Apply Filters
            </button>
          </div>
        </div>
      )}
    </div>
  );
}