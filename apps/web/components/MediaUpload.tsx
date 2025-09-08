"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileVideo, AlertCircle } from "lucide-react";
import { motion } from "framer-motion";

interface MediaUploadProps {
  onUpload: (file: File) => void;
  isLoading: boolean;
}

export function MediaUpload({ onUpload, isLoading }: MediaUploadProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (file) {
        onUpload(file);
      }
    },
    [onUpload]
  );

  const { getRootProps, getInputProps, isDragActive, fileRejections } =
    useDropzone({
      onDrop,
      accept: {
        "video/mp4": [".mp4"],
      },
      maxFiles: 1,
      maxSize: 10 * 1024 * 1024 * 1024, // 10GB
    });

  const hasRejections = fileRejections.length > 0;

  return (
    <div className="space-y-4">
      <motion.div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? "dropzone-active" : ""}`}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center space-y-4">
          <div className="p-4 bg-primary-100 rounded-full">
            {isDragActive ? (
              <Upload className="w-8 h-8 text-primary-600" />
            ) : (
              <FileVideo className="w-8 h-8 text-primary-600" />
            )}
          </div>

          <div className="text-center">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              {isDragActive ? "Drop your MP4 file here" : "Upload MP4 File"}
            </h3>
            <p className="text-gray-600 mb-4">
              Drag and drop your video file, or click to browse
            </p>
            <p className="text-sm text-gray-500">Maximum file size: 10GB</p>
          </div>

          {!isDragActive && (
            <button className="btn-primary">Choose File</button>
          )}
        </div>
      </motion.div>

      {hasRejections && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 bg-red-50 border border-red-200 rounded-lg"
        >
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-red-600" />
            <div>
              <p className="text-red-800 font-medium">File rejected</p>
              <ul className="text-red-700 text-sm mt-1">
                {fileRejections.map(({ file, errors }) => (
                  <li key={file.name}>
                    {file.name}: {errors.map((e) => e.message).join(", ")}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </motion.div>
      )}

      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-8"
        >
          <div className="inline-flex items-center space-x-2">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"></div>
            <span className="text-gray-600">Processing file...</span>
          </div>
        </motion.div>
      )}
    </div>
  );
}
