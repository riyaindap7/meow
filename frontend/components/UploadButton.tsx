"use client";

import { useState } from "react";
import { supabase } from "@/lib/supabaseClient";

export default function UploadButton() {
  const [uploading, setUploading] = useState(false);

  const uploadFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);

    const filePath = `uploads/${Date.now()}-${file.name}`;

    const { data, error } = await supabase.storage
      .from("documents")
      .upload(filePath, file);

    setUploading(false);

    if (error) {
      alert("Upload failed: " + error.message);
      return;
    }

    alert("File uploaded successfully!");
    console.log("Uploaded:", data);
  };

  return (
    <div className="border p-4 rounded-lg">
      <input
        type="file"
        accept=".pdf,.doc,.docx"
        onChange={uploadFile}
        className="cursor-pointer"
      />
      {uploading && <p className="text-sm text-gray-500 mt-2">Uploading...</p>}
    </div>
  );
}
