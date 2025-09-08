"use client";

import { DetectionMode } from "@/types";
import { motion } from "framer-motion";

interface ModeSelectorProps {
  modes: DetectionMode[];
  onSelect: (mode: DetectionMode) => void;
}

export function ModeSelector({ modes, onSelect }: ModeSelectorProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {modes.map((mode, index) => (
        <motion.div
          key={mode.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => onSelect(mode)}
          className="card cursor-pointer hover:shadow-md transition-shadow duration-200"
        >
          <div className="text-center">
            <div className="text-4xl mb-4 font-bold text-primary-600">
              {mode.name.charAt(0)}
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              {mode.name}
            </h3>
            <p className="text-gray-600">{mode.description}</p>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
