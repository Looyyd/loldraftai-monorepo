import React from "react";
import ReactDOM from "react-dom/client";
import "./styles/index.css";
import "./fonts.css";
import { ThemeProvider } from "./providers/theme-provider";
import { Layout } from "./components/Layout";
import { setStorageImpl } from "@draftking/ui/hooks/usePersistedState";

// Initialize storage implementation
setStorageImpl(window.electronAPI.storage);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <Layout />
    </ThemeProvider>
  </React.StrictMode>
);

// Use contextBridge
window.electronAPI.on("main-process-message", (_event, message) => {
  console.log(message);
});
