import { app, BrowserWindow, ipcMain, Menu } from "electron";
import { autoUpdater } from "electron-updater";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { exec } from "child_process";
import * as os from "os";
import fetch from "node-fetch";
import https from "https";
import * as fs from "fs";
import Store from "electron-store";

// const require = createRequire(import.meta.url)
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Configure auto updater
autoUpdater.logger = console;
autoUpdater.autoDownload = true;
autoUpdater.autoInstallOnAppQuit = true;

// The built directory structure
//
// ├─┬─┬ dist
// │ │ └── index.html
// │ │
// │ ├─┬ dist-electron
// │ │ ├── main.js
// │ │ └── preload.mjs
// │
process.env.APP_ROOT = path.join(__dirname, "..");

// 🚧 Use ['ENV_NAME'] avoid vite:define plugin - Vite@2.x
export const VITE_DEV_SERVER_URL = process.env["VITE_DEV_SERVER_URL"];
export const MAIN_DIST = path.join(process.env.APP_ROOT, "dist-electron");
export const RENDERER_DIST = path.join(process.env.APP_ROOT, "dist");

process.env.VITE_PUBLIC = VITE_DEV_SERVER_URL
  ? path.join(process.env.APP_ROOT, "public")
  : RENDERER_DIST;

let win: BrowserWindow | null;

interface LeagueCredentials {
  port: string;
  password: string;
}

async function getLeagueCredentials(): Promise<LeagueCredentials | null> {
  const platform = os.platform();

  if (platform === "darwin") {
    // macOS: Read lockfile from default installation path
    try {
      const lockfilePath = path.join(
        "/Applications",
        "League of Legends.app",
        "Contents",
        "LoL",
        "lockfile"
      );
      const lockfile = fs.readFileSync(lockfilePath, "utf8");
      const [, , port, password] = lockfile.split(":");
      return { port, password };
    } catch (error) {
      console.error("Error reading lockfile:", error);
      return null;
    }
  } else if (platform === "win32") {
    try {
      // First try finding the installation directory from process information
      console.log("Attempting to find League installation directory...");

      const getExecutablePath = async (): Promise<string | null> => {
        // Try with WMIC first
        try {
          const wmicResult = await new Promise<string | null>((resolve) => {
            exec(
              'wmic PROCESS WHERE name="LeagueClientUx.exe" GET ExecutablePath',
              (error, stdout) => {
                if (error || !stdout.trim()) {
                  console.log(
                    "WMIC ExecutablePath method failed:",
                    error?.message || "No output"
                  );
                  resolve(null);
                  return;
                }

                const execPath = stdout.trim().split("\n")[1]?.trim();
                if (!execPath) {
                  console.log("No executable path found in WMIC output");
                  resolve(null);
                  return;
                }

                resolve(execPath);
              }
            );
          });

          if (wmicResult) return wmicResult;
        } catch (e) {
          console.log("WMIC error:", e);
        }

        // Fallback to PowerShell
        return await new Promise<string | null>((resolve) => {
          const psCommand =
            "(Get-Process LeagueClientUx -ErrorAction SilentlyContinue).Path";
          exec(`powershell.exe -Command "${psCommand}"`, (error, stdout) => {
            if (error || !stdout.trim()) {
              console.log(
                "PowerShell Path method failed:",
                error?.message || "No output"
              );
              resolve(null);
              return;
            }

            const execPath = stdout.trim();
            if (!execPath) {
              console.log("No executable path found in PowerShell output");
              resolve(null);
              return;
            }

            resolve(execPath);
          });
        });
      };

      const execPath = await getExecutablePath();

      // The path will be something like "C:\Riot Games\League of Legends\LeagueClientUx.exe"
      // We need the directory containing the lockfile (League of Legends folder)
      let installDir = null;
      if (execPath) {
        installDir = path.dirname(execPath);
        console.log("Found League installation directory:", installDir);
      }

      // If found installation directory, try to read lockfile from there
      if (installDir) {
        const lockfilePath = path.join(installDir, "lockfile");
        if (fs.existsSync(lockfilePath)) {
          console.log("Found lockfile at:", lockfilePath);
          const lockfile = fs.readFileSync(lockfilePath, "utf8");
          const [, , port, password] = lockfile.split(":");
          return { port, password };
        }
      }

      // Fall back to checking common installation paths
      console.log("Trying common League installation paths...");
      const commonPaths = [
        path.join("C:", "Riot Games", "League of Legends", "lockfile"),
        path.join(
          process.env.LOCALAPPDATA || "",
          "Riot Games",
          "League of Legends",
          "lockfile"
        ),
        path.join("D:", "Riot Games", "League of Legends", "lockfile"),
      ];

      for (const lockfilePath of commonPaths) {
        if (fs.existsSync(lockfilePath)) {
          console.log("Found lockfile at:", lockfilePath);
          const lockfile = fs.readFileSync(lockfilePath, "utf8");
          const [, , port, password] = lockfile.split(":");
          return { port, password };
        }
      }

      // If lockfile still not found, try getting the command line arguments as last resort
      console.log("Lockfile not found, trying process command line...");

      const getCommandLine = async (): Promise<{
        port?: string;
        password?: string;
      } | null> => {
        // Try with WMIC first
        try {
          const wmicResult = await new Promise<{
            port?: string;
            password?: string;
          } | null>((resolve) => {
            exec(
              'wmic PROCESS WHERE name="LeagueClientUx.exe" GET commandline',
              (error, stdout) => {
                if (error) {
                  console.log("WMIC command line method failed:", error);
                  resolve(null);
                  return;
                }

                const port = stdout.match(/--app-port=([0-9]+)/)?.[1];
                const password = stdout.match(
                  /--remoting-auth-token=([\w-]+)/
                )?.[1];

                if (port && password) {
                  console.log("Found credentials using WMIC command line");
                  resolve({ port, password });
                } else {
                  resolve(null);
                }
              }
            );
          });

          if (wmicResult?.port && wmicResult?.password) return wmicResult;
        } catch (e) {
          console.log("WMIC error:", e);
        }

        // Fallback to PowerShell
        return await new Promise<{ port?: string; password?: string } | null>(
          (resolve) => {
            const psCommand =
              "Get-Process LeagueClientUx -ErrorAction SilentlyContinue | Select-Object CommandLine | Format-List";
            exec(`powershell.exe -Command "${psCommand}"`, (error, stdout) => {
              if (error) {
                console.log("PowerShell command line method failed:", error);
                resolve(null);
                return;
              }

              const port = stdout.match(/--app-port=([0-9]+)/)?.[1];
              const password = stdout.match(
                /--remoting-auth-token=([\w-]+)/
              )?.[1];

              if (port && password) {
                console.log("Found credentials using PowerShell command line");
                resolve({ port, password });
              } else {
                resolve(null);
              }
            });
          }
        );
      };

      const commandLineCredentials = await getCommandLine();

      if (commandLineCredentials?.port && commandLineCredentials?.password) {
        return {
          port: commandLineCredentials.port,
          password: commandLineCredentials.password,
        };
      }

      console.log("All methods failed to find League credentials");
      return null;
    } catch (error) {
      console.error("Error reading League credentials:", error);
      return null;
    }
  }

  console.error("Unsupported platform:", platform);
  return null;
}

const httpsAgent = new https.Agent({
  rejectUnauthorized: false,
});

async function getChampSelect() {
  try {
    const credentials = await getLeagueCredentials();
    if (!credentials) {
      throw new Error("League client not running or credentials not found");
    }

    const { port, password } = credentials;
    const auth = Buffer.from(`riot:${password}`).toString("base64");

    const response = await fetch(
      `https://127.0.0.1:${port}/lol-champ-select/v1/session`,
      {
        headers: {
          Authorization: `Basic ${auth}`,
        },
        agent: httpsAgent,
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error fetching champ select:", error);
    return null;
  }
}

// Register IPC handler
ipcMain.handle("get-champ-select", getChampSelect);

const store = new Store();

// Add IPC handlers for electron-store
ipcMain.handle("electron-store-get", (_event, key) => {
  return store.get(key);
});

ipcMain.handle("electron-store-set", (_event, key, value) => {
  store.set(key, value);
});

function createWindow() {
  win = new BrowserWindow({
    icon: path.join(process.env.VITE_PUBLIC, "logo512.png"),
    title: "LoLDraftAI",
    webPreferences: {
      preload: path.join(__dirname, "preload.mjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
    backgroundColor: "#09090b",
    show: false,
  });

  // Show window when ready to render
  win.once("ready-to-show", () => {
    win?.show();
    win?.maximize();
  });

  // Remove default menu
  Menu.setApplicationMenu(null);

  win.maximize();

  const downloadNotificaiton = {
    title: "LoLDraftAI",
    body: "Update available, it will automatically install when you close the app. Please wait around 30s before reopening the app to let the update complete.",
  };
  // Simplified auto-update setup
  autoUpdater.checkForUpdatesAndNotify(downloadNotificaiton);

  // Check for updates every 10 minutes
  setInterval(() => {
    autoUpdater.checkForUpdatesAndNotify(downloadNotificaiton);
  }, 10 * 60 * 1000);

  // Test active push message to Renderer-process.
  win.webContents.on("did-finish-load", () => {
    win?.webContents.send("main-process-message", new Date().toLocaleString());
  });

  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL);
  } else {
    // win.loadFile('dist/index.html')
    win.loadFile(path.join(RENDERER_DIST, "index.html"));
  }
}

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
    win = null;
  }
});

app.on("activate", () => {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.whenReady().then(createWindow);
