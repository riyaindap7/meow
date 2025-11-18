"use client";

import { useState } from "react";

export default function UploadButton() {
  const [uploading, setUploading] = useState(false);

  const uploadFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);

    // Prepare FormData for multipart upload to backend
    const formData = new FormData();
    formData.append("file", file);
    formData.append("org_id", "TEMP_ORG");
    formData.append("uploader_id", "TEMP_USER");
    formData.append("category", "uploads");

    try {
      const response = await fetch("http://localhost:8000/api/upload/upload-direct", {
        method: "POST",
        body: formData,
      });

      setUploading(false);

      if (!response.ok) {
        const errorData = await response.text();
        alert("Upload failed: " + errorData);
        return;
      }

      const data = await response.json();
      alert("File uploaded successfully!");
      console.log("Uploaded:", data);
    } catch (error) {
      setUploading(false);
      alert("Upload failed: " + (error as Error).message);
      console.error("Upload error:", error);
    }
  };

  return (
    <div className="border p-4 rounded-lg">
      <input
        type="file"
        accept=".pdf,.doc,.docx,.zip"
        onChange={uploadFile}
        className="cursor-pointer"
      />
      {uploading && <p className="text-sm text-gray-500 mt-2">Uploading...</p>}
    </div>
  );
}
