import React from "react";
import ReactDOM from "react-dom/client";
import "./styles/index.css";
import { ThemeProvider } from "./providers/theme-provider";
import { Layout } from "./components/Layout";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <Layout />
    </ThemeProvider>
  </React.StrictMode>
);

// Use contextBridge
window.ipcRenderer.on("main-process-message", (_event, message) => {
  console.log(message);
});
