"use client";

import { useAppStore } from "@/store/useAppStore";
import { Wifi, WifiOff, Settings } from "lucide-react";

export function Header() {
  const { isConnected, reset } = useAppStore();

  return (
    <header className="bg-white border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-xl font-bold text-gray-900">
              Highlight Detector
            </h1>
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <div className="flex items-center space-x-1 text-green-600">
                  <Wifi className="w-4 h-4" />
                  <span className="text-sm">Connected</span>
                </div>
              ) : (
                <div className="flex items-center space-x-1 text-red-600">
                  <WifiOff className="w-4 h-4" />
                  <span className="text-sm">Disconnected</span>
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <button className="p-2 text-gray-600 hover:text-gray-900 transition-colors">
              <Settings className="w-5 h-5" />
            </button>
            <button
              onClick={reset}
              className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              New Session
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
