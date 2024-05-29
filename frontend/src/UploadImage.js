// src/UploadImage.js
import React, { useState } from 'react';
import axios from 'axios';

const UploadImage = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [musicUrl, setMusicUrl] = useState("");

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      setLoading(true);
      const response = await axios.post("http://localhost:5000/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setMusicUrl(response.data.musicUrl);
      setLoading(false);
    } catch (error) {
      console.error("Error uploading file:", error);
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Image to Music</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} accept="image/*" />
        <button type="submit">Upload and Process</button>
      </form>
      {loading && <p>Loading...</p>}
      {musicUrl && (
        <div>
          <h2>Generated Music</h2>
          <button
            style={{
              fontSize: "2rem",
              padding: "1rem",
              marginTop: "1rem",
              backgroundColor: "#4CAF50",
              color: "white",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer",
            }}
            onClick={() => {
              const audio = new Audio(`http://localhost:5000/${musicUrl}`);
              audio.play();
            }}
          >
            Play Music
          </button>
        </div>
      )}
    </div>
  );
};

export default UploadImage;